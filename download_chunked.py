import json, os, urllib.request, concurrent.futures, time

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"
EVT = os.path.join(os.path.dirname(__file__), "data", "events")
CHUNK = 300000  # bytes per range request (stays under egress cap)

def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def validate(path):
    try:
        json.load(open(path, encoding="utf-8")); return True
    except Exception:
        return False

def download_one(mid):
    url = f"{BASE}/{mid}.json"
    path = os.path.join(EVT, f"{mid}.json")
    for attempt in range(5):
        try:
            parts, start = [], 0
            while True:
                data = fetch_range(url, start, start + CHUNK - 1)
                if not data:
                    break
                parts.append(data)
                if len(data) < CHUNK:
                    break
                start += CHUNK
            blob = b"".join(parts)
            with open(path, "wb") as f:
                f.write(blob)
            if validate(path):
                return True
        except Exception:
            time.sleep(1.0)
    return False

# gather corrupt files
bad = []
for fn in os.listdir(EVT):
    if fn.endswith(".json") and fn[0].isdigit():
        if not validate(os.path.join(EVT, fn)):
            bad.append(fn.replace(".json", ""))
print("corrupt files to fix:", len(bad))

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    res = list(ex.map(lambda mid: (mid, download_one(mid)), bad))
ok = sum(1 for _, v in res if v)
still = [m for m, v in res if not v]
print(f"fixed: {ok}/{len(bad)}  still_bad: {still}")
