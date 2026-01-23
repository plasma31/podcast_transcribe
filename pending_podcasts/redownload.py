import os
import re
import json
import time
import pathlib
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
from tqdm import tqdm

import socket
import urllib3.util.connection as urllib3_cn

urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

# ----------------------------
# CONFIG
# ----------------------------
EXCEL_PATH = "redownload list.xlsx"  # <-- your file
OUTPUT_DIR = "podcasts_downloads"
REQUEST_TIMEOUT = 25
USER_AGENT = "PodcastDownloader/1.0 (+personal-use)"
SLEEP_BETWEEN_PODCASTS_SEC = 1.0  # be polite to servers
MAX_EPISODES_PER_PODCAST = None  # set to an int to limit, e.g. 50
OVERWRITE_EXISTING = False

# If you add a column called one of these, it will be treated as "RSS feed URL"
RSS_COLUMN_CANDIDATES = [
    "RSS",
    "Rss",
    "rss",
    "Feed",
    "feed",
    "RSS Feed",
    "Feed URL",
    "RSS URL",
]

# Columns in your file that may contain a relevant page/feed URL
URL_COLUMNS_TO_TRY = ["Webseite", "Column1"]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ----------------------------
# HELPERS
# ----------------------------
def sanitize_filename(name: str, max_len: int = 200) -> str:
    name = name.strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def looks_like_feed_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return any(
        x in u for x in [".rss", "feed=", "format=rss", "/feed", "rss.xml", ".xml"]
    ) and u.startswith(("http://", "https://"))


def discover_rss_from_page(page_url: str) -> str | None:
    """Try to discover RSS/Atom feed URL from a normal webpage."""
    if not page_url or not page_url.startswith(("http://", "https://")):
        return None

    try:
        resp = session.get(page_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    content_type = (resp.headers.get("content-type") or "").lower()
    # If the page itself is already a feed (often XML)
    if "xml" in content_type or "rss" in content_type or "atom" in content_type:
        return resp.url

    soup = BeautifulSoup(resp.text, "lxml")

    # 1) Standard discovery tags
    for link in soup.find_all("link", attrs={"rel": re.compile("alternate", re.I)}):
        t = (link.get("type") or "").lower()
        href = link.get("href")
        if href and ("rss" in t or "atom" in t or "xml" in t):
            return urljoin(resp.url, href)

    # 2) Any <a href> that looks like rss/xml
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(resp.url, href)
        if looks_like_feed_url(full):
            candidates.append(full)

    # Prefer RSS over Atom if both
    for c in candidates:
        if "atom" not in c.lower():
            return c
    return candidates[0] if candidates else None


def get_feed_url(row: dict) -> str | None:
    # 1) Prefer explicit RSS column if present
    for col in RSS_COLUMN_CANDIDATES:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()

    # 2) Try URL-like columns: if value already looks like a feed, use it
    for col in URL_COLUMNS_TO_TRY:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            v = v.strip()
            if looks_like_feed_url(v):
                return v

    # 3) Try discovery from page URLs
    for col in URL_COLUMNS_TO_TRY:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            found = discover_rss_from_page(v.strip())
            if found:
                return found

    return None


def pick_episode_audio_url(entry) -> str | None:
    # feedparser puts enclosures in entry.enclosures (preferred)
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            href = enc.get("href") or enc.get("url")
            if href:
                return href

    # sometimes in links with rel="enclosure"
    if hasattr(entry, "links"):
        for l in entry.links:
            if (l.get("rel") == "enclosure") and l.get("href"):
                return l["href"]

    return None


def guess_extension_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext and len(ext) <= 6:
        return ext
    return ".mp3"


def download_file(url: str, out_path: pathlib.Path) -> None:
    if out_path.exists() and not OVERWRITE_EXISTING:
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with session.get(
        url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name[:50],
            leave=False,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                if total > 0:
                    pbar.update(len(chunk))

    tmp_path.replace(out_path)


def fetch_feed(feed_url: str) -> feedparser.FeedParserDict:
    """Fetch feed with requests to avoid urllib socket issues, then parse bytes."""
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(
                feed_url,
                timeout=(10, 30),
            )  # add headers if needed
            r.raise_for_status()

            feed = feedparser.parse(r.content)
            print(feed.feed.get("title"))
            return feed
        except Exception as e:
            last_err = e
            print(last_err)
            time.sleep(0.5 * (attempt + 1))


def main():
    df = pd.read_excel(EXCEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = df.head(1)
    # Convert NaNs to None for easier handling
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    summary = {"downloaded": 0, "skipped": 0, "podcasts": []}

    for row in records:
        podcast_name = str(row.get("Podcast Name") or "Unknown Podcast").strip()
        safe_podcast = sanitize_filename(podcast_name)
        podcast_dir = pathlib.Path(OUTPUT_DIR) / safe_podcast
        podcast_dir.mkdir(parents=True, exist_ok=True)

        feed_url = get_feed_url(row)
        if not feed_url:
            print(
                f"[SKIP] {podcast_name}: no RSS feed found (add an 'RSS' column with the feed URL)."
            )
            summary["skipped"] += 1
            summary["podcasts"].append(
                {"podcast": podcast_name, "status": "skipped_no_feed"}
            )
            continue

        print(f"\n=== {podcast_name} ===")
        print(f"Feed: {feed_url}")

        feed = fetch_feed(feed_url)
        print(feed)
        if getattr(feed, "bozo", False) and not feed.entries:
            print(f"[SKIP] {podcast_name}: failed to parse feed.")
            summary["skipped"] += 1
            summary["podcasts"].append(
                {
                    "podcast": podcast_name,
                    "status": "skipped_bad_feed",
                    "feed": feed_url,
                }
            )
            continue

        # Save feed metadata for reference
        with open(podcast_dir / "feed_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "podcast": podcast_name,
                    "feed_url": feed_url,
                    "title": getattr(feed.feed, "title", None),
                    "link": getattr(feed.feed, "link", None),
                    "updated": getattr(feed.feed, "updated", None),
                    "entries": len(feed.entries),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        entries = (
            feed.entries[:MAX_EPISODES_PER_PODCAST]
            if MAX_EPISODES_PER_PODCAST
            else feed.entries
        )

        downloaded_here = 0
        for i, entry in enumerate(entries, start=1):
            audio_url = pick_episode_audio_url(entry)
            if not audio_url:
                continue

            title = getattr(entry, "title", None) or f"episode_{i:04d}"
            safe_title = sanitize_filename(title)

            ext = guess_extension_from_url(audio_url)
            out_file = podcast_dir / f"{i:04d} - {safe_title}{ext}"

            try:
                download_file(audio_url, out_file)
                downloaded_here += 1
            except Exception as e:
                print(f"[WARN] Failed: {title} ({audio_url}) -> {e}")

        print(f"Downloaded {downloaded_here} episode file(s) for: {podcast_name}")
        summary["downloaded"] += downloaded_here
        summary["podcasts"].append(
            {
                "podcast": podcast_name,
                "status": "ok",
                "feed": feed_url,
                "downloaded_files": downloaded_here,
            }
        )

        time.sleep(SLEEP_BETWEEN_PODCASTS_SEC)

    with open(pathlib.Path(OUTPUT_DIR) / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Summary: {os.path.abspath(os.path.join(OUTPUT_DIR, '_summary.json'))}")


if __name__ == "__main__":
    main()
