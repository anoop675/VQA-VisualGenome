import os
import requests
from zipfile import ZipFile
import json
from collections import Counter
import pandas as pd

from config import ROOT, VISUAL_GENOME_BASE, QUESTION_ANS_ZIP, IMAGE_DATA_ZIP

# Utility function to download and unzip the JSON
def download_and_unzip_vg_jsons(target_dir=ROOT, timeout=60):
    os.makedirs(target_dir, exist_ok=True)  # if root dir does't exist, create directory

    qa_zip_url = f"{VISUAL_GENOME_BASE}/{QUESTION_ANS_ZIP}"  # url to the QA pairs (zip file)
    img_meta_zip_url = f"{VISUAL_GENOME_BASE}/{IMAGE_DATA_ZIP}"  # url to the image metadata (zip file)
    qa_zip_path = os.path.join(target_dir, QUESTION_ANS_ZIP)  # local target to the downloaded QA pirs (zip file)
    img_zip_path = os.path.join(target_dir, IMAGE_DATA_ZIP)  # local target to the downloaded images metadata (zip file)

    # Helper function to download a file from a URL
    def fetch(url, out_path):
        print("Downloading ZIP file from: ", url)
        fetch_request = requests.get(url, timeout=timeout,
                                     stream=True)  # send a GET request to download the file in streaming mode

        if fetch_request.status_code != 200:
            raise RuntimeError(f"Failed to download ZIP file at {url} with status {fetch_request.status_code}")

        with open(out_path, "wb") as f:
            for chunk in fetch_request.iter_content(
                    chunk_size=8192):  # write the file in chunks (to avoid loading everything into memory)
                if chunk:
                    f.write(chunk)  # only write on non-empty chunks
        print("Saved ZIP file in path: ", out_path)

    if not os.path.exists(qa_zip_path):
        fetch(qa_zip_url, qa_zip_path)
    else:
        print("Path to the QA pairs already exists in ", qa_zip_path, "\n Using QA pairs from path")

    if not os.path.exists(img_zip_path):
        fetch(img_meta_zip_url, img_zip_path)
    else:
        print("Path to the images metadata already exists in ", img_zip_path, "\n Using images metadata from path")

    # Loop over both downloaded ZIP files to extract their contents
    for z in (qa_zip_path, img_zip_path):
        print("Unzipping ZIP file: ", z)
        with ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    print("Download & unzip done, JSON files saved in ", target_dir)


# Function to load Visual Genome QA from local JSON into a flat pandas dataframe
def load_vg_qa_local(json_path, limit=2000):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        image_id = item.get("id", None)
        for qa_pair in item.get("qas", []):
            records.append({
                "image_id": image_id,
                "question": str(qa_pair["question"]).strip(),
                "answer": str(qa_pair["answer"]).strip().lower()
            })
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} QA pairs from Visual Genome")
    return df


# Load image metadata mapping
def load_vg_image_data_from_local(local_dir=ROOT):
    local_img_meta = None
    p = os.path.join(local_dir, "image_data.json")

    if os.path.exists(p):
        local_img_meta = p

    if local_img_meta is None:
        print("No image_data.json found in", local_dir, ", returing empty metadata")
        return {}

    with open(local_img_meta, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for rec in data:
        image_id = int(rec.get("image_id") or rec.get("id"))
        mapping[image_id] = {"url": rec.get("url"), "width": rec.get("width"), "height": rec.get("height")}

    print("Loaded image metadata for", len(mapping), "images")
    return mapping