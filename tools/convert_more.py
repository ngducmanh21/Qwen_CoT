import os, json
from datasets import load_dataset
from tqdm import tqdm

PARQUETS = [
    "dataset/full_vqa-00000-of-00002.parquet",
    "dataset/full_vqa-00001-of-00002.parquet",
]

OUT = "dataset/vqa_sft.jsonl"
USE_VN = True
WRITE_JSONL = False     # False nếu chỉ muốn extract ảnh

def save_image(ex):
    p = ex["image_path"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        ex["image"].save(p)

def to_record(ex):
    prompt = ex["vn_prompt"] if USE_VN else ex["en_prompt"]
    ans = "{" + ex["ground_truth"].strip() + "}"
    return {
        "id": ex["ID"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ans}]},
        ],
        "meta": {
            "subject": ex.get("subject"),
            "multiple_question": ex.get("multiple_question"),
        }
    }

def main():
    ds = load_dataset("parquet", data_files=PARQUETS, split="train")

    f = open(OUT, "w", encoding="utf-8") if WRITE_JSONL else None
    try:
        for ex in tqdm(ds, total=len(ds)):
            save_image(ex)
            if f:
                f.write(json.dumps(to_record(ex), ensure_ascii=False) + "\n")
    finally:
        if f: f.close()

    print("done")
    if WRITE_JSONL:
        print("Wrote:", OUT)

if __name__ == "__main__":
    main()
