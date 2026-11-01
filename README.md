<div align="center">

<h1>OpenNWM</h1>
<h3>A Generalist Navigation World Model<br>with Latent Action Pretraining</h3>

<p><a href="https://github.com/OpenSpace-Lab"><strong>OpenSpace Lab</strong></a></p>
<p><em>Learn navigation dynamics from action-free video. Adapt them to physical world.</em></p>

<p>
  <a href="https://openspace-lab.github.io/OpenNWM/"><img src="https://img.shields.io/badge/Project_Page-2463EB?style=flat-square&logo=githubpages&logoColor=white" alt="Project Page"></a>
  <a href="https://openspace-lab.github.io/OpenNWM/assets/paper/OpenNWM.pdf"><img src="https://img.shields.io/badge/Paper-B31B1B?style=flat-square&logo=readthedocs&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/LittleB0sS/OpenNWM"><img src="https://img.shields.io/badge/Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Models"></a>
  <a href="https://modelscope.cn/models/LittleBoss/OpenNWM"><img src="https://img.shields.io/badge/ModelScope-Models-624AFF?style=flat-square" alt="ModelScope Models"></a>
  <a href="https://huggingface.co/datasets/LittleB0sS/NavAnywhere"><img src="https://img.shields.io/badge/Data-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Data"></a>
  <a href="https://modelscope.cn/datasets/LittleBoss/NavAnywhere"><img src="https://img.shields.io/badge/ModelScope-Data-624AFF?style=flat-square" alt="ModelScope Data"></a>
</p>

<a href="website/assets/paper/method-overview-20261031.pdf">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="website/assets/paper/method-overview-dark-20261031.svg">
    <img src="website/assets/paper/method-overview-20261031.svg" width="100%" alt="OpenNWM method overview: latent action learning and navigation world modeling.">
  </picture>
</a>

<p><a href="#news">News</a> · <a href="#environment-setup">Setup</a> · <a href="#data-preparation">Data</a> · <a href="#pretrained-models">Models</a> · <a href="#evaluation">Evaluation</a> · <a href="#training">Training</a> · <a href="#todo-list">TODO List</a></p>
</div>

## News

- **2026-11-01** — Updated the paper figure and documentation. Code, models, and NavAnywhere data are available.

## Environment Setup

Requires Linux, Conda, and an NVIDIA GPU compatible with CUDA 12.4. Run from the repository root; LAM and NWM use separate environments.

```bash
git clone https://github.com/OpenSpace-Lab/OpenNWM.git
cd OpenNWM
for module in lam nwm; do
  conda create -n "open-$module" python=3.10 -y
  conda run -n "open-$module" python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  conda run -n "open-$module" python -m pip install -r "$module/requirements.txt"
done

conda run -n open-nwm python setup_paths.py \
  --data-root /data/navigation --work-root /data/opennwm --nav-profile 15 \
  --lam-python "$(conda run -n open-lam python -c 'import sys; print(sys.executable)')" \
  --nwm-python "$(conda run -n open-nwm python -c 'import sys; print(sys.executable)')"
source .env
```

Replace `/data/navigation` and `/data/opennwm` with your data and work directories. Run `source .env` in each new terminal. To reuse existing splits, add `--split-root /path/to/splits` to `setup_paths.py`.

## Data Preparation

Download NavAnywhere from [Hugging Face](https://huggingface.co/datasets/LittleB0sS/NavAnywhere) or [ModelScope](https://modelscope.cn/datasets/LittleBoss/NavAnywhere) and extract it to `$NWM_DATA_ROOT`. The archives total **1.93 TB**; extraction requires `tar`, `zstd`, and additional disk space.

See [DATA.md](DATA.md) for extraction and directory layout. **TartanDrive requires a separate download** for world model post-training and default LAM evaluation.

## Pretrained Models

Download from [Hugging Face](https://huggingface.co/LittleB0sS/OpenNWM) or [ModelScope](https://modelscope.cn/models/LittleBoss/OpenNWM) into `$OPENNWM_WORK_ROOT/models/OpenNWM`, including `configs/`.

| Checkpoint | Use |
| :--- | :--- |
| `lam-nav15-pixel-action-step60k.ckpt` | Latent action extraction |
| `opennwm-nav15-stage1-pretrain60k.pth.tar` | World model pretraining checkpoint |
| `opennwm-nav15-stage2-joint100k.pth.tar` | Physical-action prediction and planning |

## Evaluation

After preparing test data, import the Stage 2 checkpoint and run prediction:

```bash
"$NWM_PYTHON" run.py nwm import-checkpoint \
  --checkpoint "$OPENNWM_WORK_ROOT/models/OpenNWM/opennwm-nav15-stage2-joint100k.pth.tar" \
  --config "$OPENNWM_WORK_ROOT/models/OpenNWM/configs/opennwm-nav15-stage2.yaml" \
  --run-dir "$OPENNWM_WORK_ROOT/runs/released-stage2"
"$NWM_PYTHON" run.py nwm infer \
  exp_dir="$OPENNWM_WORK_ROOT/runs/released-stage2" ckp=latest eval_type=time \
  prediction_dir="$OPENNWM_WORK_ROOT/eval/stage2/pred"
```

Metrics, rollouts, and planning: [NWM evaluation](nwm/README.md#评测). Latent action reconstruction: [LAM evaluation](lam/README.md#正式评测).

## Training

Follow [LAM training](lam/README.md#训练与续跑) → [cache preparation](nwm/README.md#准备缓存) → [world model pretraining and adaptation](nwm/README.md#训练). Use the NavAnywhere 15-source profile and matching LAM checkpoints for caches.

## TODO List

- [x] Release code, data, and pretrained models.
- [x] Publish the paper, project page, and demos.
- [x] Provide training and evaluation guides.
- [ ] Support closed-loop navigation on Unitree Go2 ([deployment plan](GO2_DEPLOYMENT.md)).

## Acknowledgements & Citation

We thank [NWM](https://github.com/facebookresearch/nwm#bibtex), [CompACT](https://github.com/kdwonn/CompACT#citation), [DreamDojo](https://github.com/NVIDIA/DreamDojo#-citation), and the dataset authors.

If you use OpenNWM, please cite:

```bibtex
@unpublished{opennwm,
  title = {OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining},
  author = {Nie, Dujun and Song, Ruiqi and Wang, Zixuan and Wang, Hanqing and Han, Yifan and Ding, Baiyong and Zheng, Wenzhao and Wu, Hangbin and Chen, Long},
  note = {Under review as a conference paper at ICLR 2027}
}
```

NWM-derived components retain [CC BY-NC 4.0](licenses/NWM-CC-BY-NC-4.0.txt); DreamDojo LAM components retain [Apache 2.0](licenses/DreamDojo-Apache-2.0.txt). Datasets and external models retain their own licenses.
