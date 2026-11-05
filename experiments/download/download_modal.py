import os
import modal

app = modal.App("mssearch-downloader")

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "wget", "tar")
    .pip_install("huggingface_hub", "requests", "tqdm")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)

@app.function(
    image=download_image,
    volumes={"/data": data_volume, "/models": models_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_remote_assets(
    download_dataset: bool = True,
    download_aux: bool = True,
    model_name: str = "Qwen2.5-VL-7B-Instruct",
    dataset_url: str = None,
    images_url: str = None,
):
    import urllib.request
    import tarfile
    import zipfile
    from huggingface_hub import snapshot_download

    # 1. Dataset
    if download_dataset:
        print("[Modal] Setting up STASHBench dataset in /data...")
        os.makedirs("/data/STASHBench/pool", exist_ok=True)
        if dataset_url:
            print(f"[Modal] Downloading dataset annotations from {dataset_url}...")
            urllib.request.urlretrieve(dataset_url, "/data/STASHBench/annotations_part1234_merged.json")
        if images_url:
            print(f"[Modal] Downloading image archive from {images_url}...")
            zip_tmp = "/data/STASHBench/pool.zip"
            urllib.request.urlretrieve(images_url, zip_tmp)
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                zf.extractall("/data/STASHBench/pool")
            if os.path.exists(zip_tmp):
                os.remove(zip_tmp)
        data_volume.commit()

    # 2. Auxiliary models
    if download_aux:
        print("[Modal] Setting up auxiliary models in /models...")
        os.makedirs("/models/sam3", exist_ok=True)
        os.makedirs("/models/spacy", exist_ok=True)
        sam3_url = "https://www.modelscope.cn/models/llp1995/sam3/resolve/master/sam3.tar.gz"
        spacy_url = "https://www.modelscope.cn/models/llp1995/sam3/resolve/master/en_core_web_sm-3.8.0.tar.gz"
        
        sam3_tar = "/models/sam3.tar.gz"
        if not os.path.exists("/models/sam3/sam3.pt"):
            urllib.request.urlretrieve(sam3_url, sam3_tar)
            with tarfile.open(sam3_tar, "r:*") as tar:
                tar.extractall("/models/sam3")
            if os.path.exists(sam3_tar):
                os.remove(sam3_tar)

        spacy_tar = "/models/spacy.tar.gz"
        if not os.path.exists("/models/spacy/en_core_web_sm-3.8.0"):
            urllib.request.urlretrieve(spacy_url, spacy_tar)
            with tarfile.open(spacy_tar, "r:*") as tar:
                tar.extractall("/models/spacy")
            if os.path.exists(spacy_tar):
                os.remove(spacy_tar)

        # SigLIP
        snapshot_download(
            repo_id="google/siglip2-so400m-patch16-512",
            local_dir="/models/siglip2-so400m-patch16-512",
            local_dir_use_symlinks=False
        )
        models_volume.commit()

    # 3. Target Model
    if model_name:
        from download_model import MODEL_MAP
        hf_repo = MODEL_MAP.get(model_name, model_name)
        target_dir = f"/models/{model_name}"
        print(f"[Modal] Downloading {model_name} ({hf_repo}) to {target_dir}...")
        snapshot_download(
            repo_id=hf_repo,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN")
        )
        models_volume.commit()
    print("[Modal] Download complete. Volumes committed.")

@app.local_entrypoint()
def main(
    model: str = "Qwen2.5-VL-7B-Instruct",
    dataset: bool = True,
    aux: bool = True,
    dataset_url: str = None,
    images_url: str = None,
):
    print(f"Launching Modal Asset Downloader for model: {model}")
    download_remote_assets.remote(
        download_dataset=dataset,
        download_aux=aux,
        model_name=model,
        dataset_url=dataset_url,
        images_url=images_url
    )
