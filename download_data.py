import json, os, urllib.request, time

COMP = 43
SEASON = 3
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(__file__), "data")
EVT_DIR = os.path.join(OUT, "events")
os.makedirs(EVT_DIR, exist_ok=True)

def fetch(url, tries=5):
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = e
            print(f"  retry {t+1}/{tries} for {url}: {e}")
            time.sleep(1.5 + t)
    raise last

# matches
murl = f"{BASE}/matches/{COMP}/{SEASON}.json"
print("fetching matches:", murl)
matches = fetch(murl)
with open(os.path.join(OUT, f"matches_{COMP}_{SEASON}.json"), "w") as f:
    json.dump(matches, f, ensure_ascii=False, indent=2)
print(f"  -> {len(matches)} matches")

ids = [m["match_id"] for m in matches]
print("downloading events for", len(ids), "matches...")
ok = 0
for i, mid in enumerate(ids):
    path = os.path.join(EVT_DIR, f"{mid}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        ok += 1
        continue
    try:
        evt = fetch(f"{BASE}/events/{mid}.json")
        with open(path, "w") as f:
            json.dump(evt, f, ensure_ascii=False)
        ok += 1
    except Exception as e:
        print("  FAILED", mid, e)
    if (i + 1) % 10 == 0:
        print(f"  progress {i+1}/{len(ids)} (ok={ok})")
        time.sleep(0.3)
print(f"DONE. events downloaded: {ok}/{len(ids)}")
