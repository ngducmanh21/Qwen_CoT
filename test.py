from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files=[
        "dataset/full_vqa-00000-of-00002.parquet",
        "dataset/full_vqa-00001-of-00002.parquet",
    ],
    split="train",
)
print(ds.column_names)
print(ds[0])
