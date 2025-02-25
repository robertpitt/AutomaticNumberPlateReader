rm -rf train/*

tesstrain.sh --fonts_dir fonts \
  --fontlist "Charles Wright Bold" \
  --lang eng \
  --linedata_only \
  --langdata_dir langdata_lstm \
  --tessdata_dir tesseract/tessdata \
  --save_box_tiff \
  --maxpages 10 \
  --output_dir train