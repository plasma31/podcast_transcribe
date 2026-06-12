import os
import json
import time
import requests
import pandas as pd
from pathlib import Path

# === CONFIG ===
ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "data_sources" / "list.xlsx"
OUTPUT_DIR = ROOT / "fyyd_downloads"
RESULTS_JSON = ROOT / "artifacts" / "acquisition" / "fyyd_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

FYYD_BASE = "https://api.fyyd.de/0.2"

# === FUNCTIONS ===
def safe_request(url, retries=3):
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=(10,20), stream=True)
        except Exception as e:
            print(f"Retrying ({attempt+1}/{retries}) after error: {e}")
            time.sleep(3)
    raise Exception(f"Failed after {retries} retries")

def fyyd_search_podcast(name):
    """Search podcast by name using fyyd API."""
    url = f"{FYYD_BASE}/search/podcast"

    resp = requests.get(url, params={"term": name})
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", [])
    if not results:
        return None
    # Return first matching podcast (you can improve matching logic)
    return results[0]


def fyyd_get_episodes(podcast_id):
    """Fetch episodes for a podcast id."""
    url = f"{FYYD_BASE}/podcast/episodes"
    resp = requests.get(url, params={"podcast_id": podcast_id, "count":1000})
    resp.raise_for_status()
    episodes_data = resp.json().get("data",[])
    episodes = episodes_data.get("episodes", [])
    return episodes


def download_audio(filename, url, output_dir):
    """Download audio file safely using streaming, so it won't stall."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    try:
        with requests.get(url, stream=True, timeout=(10, 20)) as r:
            # (connect timeout = 10 sec, read timeout = 20 sec)
            r.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):  # 256 KB chunks
                    if chunk:
                        f.write(chunk)

        return save_path

    except requests.exceptions.Timeout:
        raise Exception("Download timed out (server too slow or stalled).")

    except Exception as e:
        raise Exception(f"Download failed: {e}")


def download_with_retries(filename, url, output_dir, retries=3):
    for i in range(retries):
        try:
            return download_audio(filename, url, output_dir)
        except Exception as e:
            print(f"Retry {i+1}/{retries} failed: {e}")
            time.sleep(2)

    raise Exception("All retries failed")


# === MAIN ===

df = pd.read_excel(EXCEL_PATH)
results = []
for idx, row in df.iterrows():
    record = row.to_dict()
    record["success"] = False
    record["podcast_id"] = None
    record["episode_count"] = 0
    record["episodes_downloaded"] = 0
    record["error"] = None
    record["failed_ep"] = []
    record["downloaded"]  = []
    podcast_name = str(row.get("Podcast Name") or row.get("name") or "").strip()
    if not podcast_name:
        record["error"] = "No podcast name found"
        results.append(record)
        continue

    try:
        print(f"\n[{idx}] Searching podcast: {podcast_name}")
        podcast = fyyd_search_podcast(podcast_name)
        if not podcast:
            record["error"] = "No podcast found on fyyd"
            results.append(record)
            continue

        podcast_id = podcast["id"]
        record["podcast_id"] = podcast_id
        # Print podcast_name
        episodes = fyyd_get_episodes(podcast_id)
        record["episode_count"] = len(episodes)

        if not episodes:
            record["error"] = "No episodes found"
            results.append(record)
            continue

        # Download first episode as demo (or all)
        downloaded = 0

        for ep in episodes:
            ep_title = ep.get("title")
            audio_url = ep.get("enclosure")
            episode_number = ep.get("num_episode")
            if not audio_url:
                continue
            try:
                filename = os.path.basename(ep_title+"-"+str(episode_number)+".mp3")
                save_path = download_with_retries(filename,audio_url, OUTPUT_DIR / podcast_name)
                downloaded += 1
                print(f"✅ Downloaded {ep_title}")
                downloaded_obj = {"title":ep_title,  "id" : ep.get("id")}
                record["downloaded"].append(downloaded_obj)
            except Exception as e:
                print(f"⚠️ Failed to download {ep_title}: {e}")
                failed_obj = { "title" : ep_title, "id" : ep.get("id")}
                print(failed_obj)
                record["failed_ep"].append(failed_obj)
            time.sleep(3)  # be nice to API

        record["episodes_downloaded"] = downloaded
        record["success"] = downloaded > 0

    except Exception as e:
        print(e)
        record["error"] = str(e)
        print(f"❌ Error: {e}")

    results.append(record)

# Save results
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== Summary ===")
print(f"Total rows: {len(results)}")
print(f"Successful: {sum(r['success'] for r in results)}")
print(f"Results saved to {RESULTS_JSON}")
