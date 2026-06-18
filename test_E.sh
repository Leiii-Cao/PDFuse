SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export RESULT="./result/"

python PDFuse.py \
  --image1 "./data/PDFuse_E/input1/250279.jpg" \
  --image2 "./data/PDFuse_E/input2/250279.jpg" \
  --output "$RESULT" \
  --degradation1 "gaussian" \
  --degradation2 "thermal_diffusion"
