"""Download a subset of Argoverse 2 Motion Forecasting from the public S3 bucket.

No account / credentials needed. Each scenario = 1 parquet (tracks) + 1 json (vector map).

    python scripts/download_av2.py --split train --num 20000 --out data/av2
    python scripts/download_av2.py --split val   --num 2000  --out data/av2
"""
import argparse
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

BUCKET = "https://argoverse.s3.amazonaws.com"
PREFIX = "datasets/av2/motion-forecasting/{split}/"


def list_scenarios(split: str, limit: int):
    """List scenario ids using S3 ListObjectsV2 with delimiter (one CommonPrefix per scenario)."""
    ids, token = [], None
    sess = requests.Session()
    while len(ids) < limit:
        params = {"list-type": "2", "prefix": PREFIX.format(split=split), "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = sess.get(BUCKET + "/?" + urllib.parse.urlencode(params), timeout=60)
        r.raise_for_status()
        ids += re.findall(r"<Prefix>[^<]*/" + split + r"/([0-9a-f\-]+)/</Prefix>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return ids[:limit]


def fetch(sess, url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    for attempt in range(5):
        try:
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
            return
        except Exception:
            if attempt == 4:
                raise


def download_one(sess, split, sid, out):
    d = os.path.join(out, split, sid)
    os.makedirs(d, exist_ok=True)
    base = f"{BUCKET}/{PREFIX.format(split=split)}{sid}/"
    fetch(sess, base + f"scenario_{sid}.parquet", os.path.join(d, f"scenario_{sid}.parquet"))
    fetch(sess, base + f"log_map_archive_{sid}.json", os.path.join(d, f"log_map_archive_{sid}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=20000)
    ap.add_argument("--out", default="data/av2")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ids", default=None, help="file with scenario ids (e.g. splits/train_ids.txt) for an exact re-download")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else list_scenarios(a.split, a.num)
    print(f"{a.split}: {len(ids)} scenarios")
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=a.workers, pool_maxsize=a.workers)
    sess.mount("https://", adapter)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(download_one, sess, a.split, s, a.out) for s in ids]
        for f in tqdm(as_completed(futs), total=len(futs)):
            f.result()


if __name__ == "__main__":
    main()
