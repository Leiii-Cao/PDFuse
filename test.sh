## Infrared-Visible Image Fusion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export RESULT="./result/"

export DATA_DIR="./data/IVF/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task IVF

## Multi-Exposure Image Fusion
export DATA_DIR="./data/MEIF/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MEIF

## Multi-Focus Image Fusion
export DATA_DIR="./data/MFIF/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MFIF

## Multi-Modal Medical Image Fusion
export DATA_DIR="./data/MMIF/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MMIF
