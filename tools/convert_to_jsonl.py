import json
from datasets import load_dataset
from tqdm import tqdm

PARQUETS = [
    "dataset/full_vqa-00000-of-00002.parquet",
    "dataset/full_vqa-00001-of-00002.parquet",
]

OUT = "dataset/vqa_sft.jsonl"
USE_VN = True
USE_IMAGE_PATH = True

# optional: ghi 2 dòng/1 sample (VI + EN)
BILINGUAL_DUP = True  # True -> output both vi & en rows

def make_user_content(ex, prompt: str):
    if not USE_IMAGE_PATH:
        raise ValueError("Không serialize PIL vào jsonl được. Dùng image_path nhé.")
    return [
        {"type": "image", "image": ex["image_path"]},
        {"type": "text", "text": prompt},
    ]

def to_record(ex, lang: str):
    prompt = ex["vn_prompt"] if lang == "vn" else ex["en_prompt"]
    user_content = make_user_content(ex, prompt)

    answer = (ex["ground_truth"] or "").strip().upper()
    assistant_text = "{" + answer + "}"

    return {
        "id": ex["ID"],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {
            "lang": lang,
            "subject": ex.get("subject"),
            "multiple_question": ex.get("multiple_question"),
        }
    }

def main():
    ds = load_dataset("parquet", data_files=PARQUETS, split="train")
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, total=len(ds)):
            if BILINGUAL_DUP:
                f.write(json.dumps(to_record(ex, "vn"), ensure_ascii=False) + "\n")
                f.write(json.dumps(to_record(ex, "en"), ensure_ascii=False) + "\n")
            else:
                lang = "vn" if USE_VN else "en"
                f.write(json.dumps(to_record(ex, lang), ensure_ascii=False) + "\n")
    print("Wrote:", OUT)

if __name__ == "__main__":
    main()
