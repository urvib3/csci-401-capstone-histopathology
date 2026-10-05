# Download and extract the Kaggle BCSS release (~8.3 GB zip) into data/bcss.
#
# The dataset is public (CC0), so Kaggle serves it without credentials.
# Needs ~17 GB free while the zip and extracted files coexist; pass
# --keep-zip to keep the archive afterwards.
#
#   python3 scripts/download_bcss.py [--out data/bcss]

import argparse
import os
import shutil
import urllib.error
import urllib.request
import zipfile

URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "whats2000/breast-cancer-semantic-segmentation-bcss"
)


def download(url, path):
    # Resume a partial download if one is present.
    done = os.path.getsize(path) if os.path.exists(path) else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 416:  # nothing left to fetch
            print("Already downloaded.")
            return
        raise
    with response:
        if response.status != 206:
            done = 0  # server ignored the range; start over
        total = done + int(response.headers.get("Content-Length", 0))
        with open(path, "ab" if done else "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                print(f"\r{done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download the Kaggle BCSS dataset.")
    parser.add_argument("--out", default=os.path.join("data", "bcss"))
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(os.path.join(args.out, "BCSS_512")):
        print(f"{args.out} already contains the dataset.")
        return

    zip_path = args.out.rstrip("/") + ".zip"
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    print(f"Downloading to {zip_path}")
    download(URL, zip_path)

    print(f"Extracting to {args.out}")
    tmp = args.out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp)
    os.replace(tmp, args.out)
    if not args.keep_zip:
        os.remove(zip_path)

    for sub in ("BCSS", "BCSS_512"):
        path = os.path.join(args.out, sub)
        print(f"{path}: {'ok' if os.path.isdir(path) else 'MISSING'}")


if __name__ == "__main__":
    main()
