# MSSearch Asset Download Suite

Unified download and asset preparation tools for evaluation datasets, auxiliary perception weights (SAM3, SigLIP, spaCy), and vision-language foundation backbones.

---

## Component Scripts

1. `download_data.py`:
   - Validates or downloads `200_samples.json`.
   - Populates and checks the image repository directory `<DATA_DIR>/images`.
   - Usage:
     ```bash
     python download_data.py --data_dir /data/shared/cvpr/dat/simple/data
     ```

2. `download_auxiliary.py`:
   - Downloads official SAM3 checkpoint (`sam3.pt`) and spaCy pipeline (`en_core_web_sm-3.8.0`) from ModelScope mirrors.
   - Downloads `google/siglip-base-patch16-384` from Hugging Face.
   - Usage:
     ```bash
     python download_auxiliary.py --base_model_dir /data/shared/cvpr/dat/models
     ```

3. `download_model.py`:
   - Downloads individual model checkpoints by name or HuggingFace repo ID:
     ```bash
     python download_model.py --model_name Qwen2.5-VL-7B-Instruct --base_model_dir /data/shared/cvpr/dat/models
     python download_model.py --model_name Yi-VL-34B --base_model_dir /data/shared/cvpr/dat/models
     python download_model.py --model_name favor123/llava-hr-7b-sft-1024 --base_model_dir /data/shared/cvpr/dat/models
     ```

4. `download_all.py` (Unified CLI Orchestrator):
   - Sequentially runs data, auxiliary, and model downloads.
   - Flags: `--all_baseline` (downloads all 17 baseline models), `--all_our` (downloads 8 MSSearch models), or `--models <name1> <name2>`.

5. `download_modal.py` (Modal Cloud Storage Population):
   - Downloads assets directly into Modal Volumes (`mssearch-data` and `mssearch-models`).
   - Usage:
     ```bash
     modal run download_modal.py --model Qwen2.5-VL-7B-Instruct
     ```
