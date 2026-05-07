import os
import shutil
from huggingface_hub import hf_hub_download
from datasets import load_dataset

root = os.path.join(os.environ["LOCAL_DATASETS_DIR"], "drop")
tmp = root + ".rebuild"
cache_dir = os.path.join(os.environ["LOCAL_DATASETS_DIR"], "_hf_cache_drop_manual")

for p in [tmp, cache_dir]:
    if os.path.exists(p):
        shutil.rmtree(p)

data_files = {
    "train": hf_hub_download(
        repo_id="ucinlp/drop",
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
        cache_dir=cache_dir,
    ),
    "validation": hf_hub_download(
        repo_id="ucinlp/drop",
        repo_type="dataset",
        filename="data/validation-00000-of-00001.parquet",
        cache_dir=cache_dir,
    ),
}

ds = load_dataset("parquet", data_files=data_files)
print(ds)
ds.save_to_disk(tmp)
print("saved to", tmp)