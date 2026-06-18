export RESULT="./result/"

python PDFuse.py \
  --image1 "/data1/Caolei/PDFuse_Finnal/data/PDFuse_E/input1/250279.jpg" \
  --image2 "/data1/Caolei/PDFuse_Finnal/data/PDFuse_E/input2/250279.jpg" \
  --output "$RESULT" \
  --degradation1 "gaussian" \
  --degradation2 "thermal_diffusion"
