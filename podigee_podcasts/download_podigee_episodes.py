import os
import json
import time
import requests
import pandas as pd

# === CONFIG ===
EXCEL_PATH = "podigee_episodes.csv"
OUTPUT_DIR = "fyyd_downloads"
RESULTS_JSON = "podigee.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

df = pd.read_csv(EXCEL_PATH, encoding_errors="replace")

results = []

currRecord = None
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
    if(currRecord is None):
        currRecord = record
        
    if not podcast_name:
        record["error"] = "No podcast name found"
        results.append(record)
        continue
         
    try:
        if(currRecord["Podcast Name"] is not podcast_name):
            results.append(currRecord)
            currRecord =record
       
        ep_title = row.get("title")
        audio_url = row.get("mp3_url")
        if not audio_url:
            continue
        try:
            filename = os.path.basename(ep_title+"-"+".mp3")
            currRecord["episodes_downloaded"] += 1
            print(f"✅ Downloaded {ep_title}")
            downloaded_obj = {"title":ep_title}
            currRecord["downloaded"].append(downloaded_obj)
        except Exception as e:
            print(f"⚠️ Failed to download {ep_title}: {e}")
            failed_obj = { "title" : ep_title}
            print(failed_obj)
            currRecord["failed_ep"].append(failed_obj)
        time.sleep(3)  # be nice to API

        record["success"] = currRecord["episodes_downloaded"]  > 0

    except Exception as e:
        print(e)
        record["error"] = str(e)
        print(f"❌ Error: {e}")
    
# Save results
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== Summary ===")
print(f"Total rows: {len(results)}")
print(f"Successful: {sum(r['success'] for r in results)}")
print(f"Results saved to {RESULTS_JSON}")
