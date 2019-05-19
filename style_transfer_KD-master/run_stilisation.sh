python neural_style/neural_style.py eval \
  --content-image images/content-images/amber.jpg \
  --model distillation_results/checkpoint_0.tar \
  --output-image images/output-images/test2.jpg \
  --cuda 1 \
  --distilled
  
#--model saved_models/mosaic.pth \
