## Infrared-Visible Image Fusion
export DATA_DIR="/data1/Caolei/PDFuse_Finnal/data/IVF/"
export RESULT="./result/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task IVF

## Multi-Exposure Image Fusion
export DATA_DIR="/data1/Caolei/PDFuse_Finnal/data/MEIF/"
export RESULT="./result/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MEIF

## Multi-Focus Image Fusion
export DATA_DIR="/data1/Caolei/PDFuse_Finnal/data/MFIF/"
export RESULT="./result/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MFIF

## Multi-Modal Medical Image Fusion
export DATA_DIR="/data1/Caolei/PDFuse_Finnal/data/MMIF/"
export RESULT="./result/"
CUDA_VISIBLE_DEVICES=0 python PDFuse.py --fusion_task MMIF
