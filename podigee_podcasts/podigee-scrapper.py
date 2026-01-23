#!/usr/bin/env python3
"""
Podigee podcast MP3 scraper
- Given a list of Podigee subdomains (or full Podigee URLs), tries to collect episode MP3 URLs.
- Outputs CSV: podcast_name, episode_title, pub_date, mp3_url, episode_page
"""

import requests
import time
import csv
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

HEADERS = {"User-Agent": "podigee-scraper/1.0 (+contact: your-email@example.com)"}
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds

def normalize_to_podigee_domain(s):
    # Accept: 'systemsprenger', 'systemsprenger.podigee.io', 'https://systemsprenger.podigee.io'
    s = s.strip()
    if s.startswith("http"):
        p = urlparse(s)
        host = p.netloc
        if host.endswith("podigee.io"):
            return p.scheme + "://" + host
        # if path contains podigee, keep full
        return s.rstrip("/")
    if s.endswith("podigee.io"):
        return "https://" + s
    return "https://%s.podigee.io" % s

def try_feed_mp3(base_url):
    # Podigee often exposes /feed/mp3
    urls = [
        urljoin(base_url + "/", "feed/mp3"),
        urljoin(base_url + "/", "feed"),
        urljoin(base_url + "/", "feed/rss"),
        base_url
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=HEADERS, timeout=20)
        except Exception as e:
            continue
        if r.status_code != 200:
            continue
        content_type = r.headers.get("Content-Type","")
        text = r.text
        # If this looks like RSS / XML, parse for <enclosure url="...mp3">
        if ("xml" in content_type) or text.lstrip().startswith("<?xml") or "<rss" in text[:2000].lower():
            try:
                root = ET.fromstring(text.encode('utf-8', errors='ignore'))
                items = root.findall(".//item")
                results = []
                for it in items:
                    title = (it.findtext("title") or "").strip()
                    pub = (it.findtext("pubDate") or it.findtext("date") or "").strip()
                    # find enclosure
                    enc = it.find("enclosure")
                    if enc is not None and enc.get("url"):
                        mp3 = enc.get("url")
                        results.append((title, pub, mp3, u))
                    else:
                        # sometimes <link> holds a page with mp3
                        link = it.findtext("link") or ""
                        results.append((title, pub, None, link))
                if results:
                    return results, u
            except Exception:
                pass
        # If HTML page, we can look for audio tag or audio.podigee-cdn links
        if "text/html" in content_type or "<html" in text.lower():
            # quick search for podigee-cdn mp3 urls
            found = re.findall(r"https?://audio\.podigee-cdn\.net/[^\s\"'<>]+\.mp3[^\s\"'<>]*", text)
            if found:
                res=[]
                for f in found:
                    # best-effort: try to infer title from surrounding <title> tag
                    title = re.search(r"<title>(.*?)</title>", text, re.I|re.S)
                    t = title.group(1).strip() if title else ""
                    res.append((t, "", f, u))
                return res, u
    return None, None

def scrape_podcast(podigee_input):
    base = normalize_to_podigee_domain(podigee_input)
    print("→ probing", base)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    items, feed_url = try_feed_mp3(base)
    if not items:
        print("  - no feed/mp3 found or feed couldn't be parsed for", base)
        return []
    out=[]
    for title, pub, mp3, page in items:
        if mp3 is None:
            # try to fetch page and find an mp3
            try:
                r = requests.get(page, headers=HEADERS, timeout=15)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                found = re.search(r"https?://audio\.podigee-cdn\.net/[^\s\"'<>]+\.mp3[^\s\"'<>]*", r.text)
                mp3 = found.group(0) if found else None
            except Exception:
                mp3 = None
        out.append({"podcast": base, "title": title, "pub_date": pub, "mp3_url": mp3, "episode_page": page, "feed_url": feed_url})
    return out

def main():
    # Edit this list with the Podigee hosts you want to scrape.
    podcast_inputs = [
        "ost-sa-fachgespraeche",
        "ideequadrat",
        "erdbeerfroescheundteppichaepfel"
    ]

    all_rows = []
    for p in podcast_inputs:
        try:
            rows = scrape_podcast(p)
            all_rows.extend(rows)
        except Exception as e:
            print("Error scraping", p, e)

    # save CSV
    fname = "podigee_episodes.csv"
    keys = ["podcast","title","pub_date","mp3_url","episode_page","feed_url"]
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("Saved", fname)

if __name__ == "__main__":
    main()
