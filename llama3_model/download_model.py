import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B",
    local_dir="/capsule/home/xiangyuxing/hf_offline/Llama-3.2-3B",
    token=os.environ["HF_TOKEN"],
    local_dir_use_symlinks=False,
    resume_download=True,
)