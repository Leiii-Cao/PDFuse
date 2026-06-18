# [NeurIPS 2025] Projection-Manifold Regularized Latent Diffusion for Robust General Image Fusion

This repository is the official implementation of the **NeurIPS 2025** paper:
_"Projection-Manifold Regularized Latent Diffusion for Robust General Image Fusion"_.

### [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7921c529626c3ef26cc3308c6b73f070-Abstract-Conference.html) | [Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/117987)

![PDFuse overview](assets/fig1.jpg)

## News

- Code and demo data are released.
- The pretrained Stable Diffusion weights are not included in this repository. Please prepare them locally before running inference.

## Installation

Clone this repository:

```bash
git clone https://github.com/Leiii-Cao/PDFuse.git
cd PDFuse
```

Create and activate the conda environment:

```bash
conda create -n PDFuse python=3.11.5
conda activate PDFuse
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Pretrained Model

PDFuse uses Stable Diffusion v1.5 as the latent diffusion prior. Please download the Stable Diffusion v1.5 model and place it under:

```text
pretrained/stable-diffusion-v1-5/
```

The default model path used by the code is:

```bash
./pretrained/stable-diffusion-v1-5/
```

You can also specify a custom model path with:

```bash
python PDFuse.py --model_id /path/to/stable-diffusion-v1-5
```

## Data Structure

For paired image fusion, each task folder should contain two subfolders with paired images using the same filenames. Example:

```text
data/
  IVF/
    input1/
      230164.jpg
    input2/
      230164.jpg
  MEIF/
    under/
    over/
  MFIF/
    near/
    far/
  MMIF/
    ct/
    mar/
```

The repository includes several demo image pairs under `data/`.

## Inference

### Run Demo Tasks

Infrared-visible image fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python PDFuse.py \
  --data_dir ./data/IVF \
  --output ./result \
  --fusion_task IVF
```

Multi-exposure image fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python PDFuse.py \
  --data_dir ./data/MEIF \
  --output ./result \
  --fusion_task MEIF
```

Multi-focus image fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python PDFuse.py \
  --data_dir ./data/MFIF \
  --output ./result \
  --fusion_task MFIF
```

Multi-modal medical image fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python PDFuse.py \
  --data_dir ./data/MMIF \
  --output ./result \
  --fusion_task MMIF
```

### Run One Image Pair

```bash
CUDA_VISIBLE_DEVICES=0 python PDFuse.py \
  --image1 ./data/PDFuse_E/input1/250279.jpg \
  --image2 ./data/PDFuse_E/input2/250279.jpg \
  --output ./result \
  --degradation1 gaussian \
  --degradation2 thermal_diffusion
```

### CPU Mode

GPU inference is recommended. For a small test on CPU, add `--cpu`:

```bash
python PDFuse.py \
  --image1 ./data/PDFuse_E/input1/250279.jpg \
  --image2 ./data/PDFuse_E/input2/250279.jpg \
  --output ./result \
  --cpu
```

## Arguments

Common arguments:

- `--data_dir`: directory containing two paired image subfolders.
- `--image1`, `--image2`: paths for one-pair inference.
- `--output`: directory for fused images and observed inputs.
- `--model_id`: path to Stable Diffusion v1.5.
- `--fusion_task`: one of `IVF`, `MEIF`, `MFIF`, or `MMIF`.
- `--degradation1`, `--degradation2`: degradation settings for the two source images.
- `--nfe`: number of function evaluations / sampling steps.
- `--cpu`: run on CPU.

Supported degradation examples include `none`, `lowlight`, `overexposure`, `gaussian`, `motion`, and `thermal_diffusion`.

## Output

Results are saved to the output directory, for example:

```text
result/
  250279.jpg
  observed1/
  observed2/
```

## Citation

If our work assists your research, feel free to give us a star or cite us using:

```bibtex
@article{cao2026projection,
  title={Projection-Manifold Regularized Latent Diffusion for Robust General Image Fusion},
  author={Cao, Lei and Zhang, Hao and Li, Chunyu and Ma, Jiayi},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={83972--84003},
  year={2026}
}
```

## Contact

If you have any questions or discussions, please send me an email:

```text
whu.caolei@whu.edu.cn
```
