CUDA_VISIBLE_DEVICES=3 torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/seg/doublebell_viz.yaml \
  --checkpoint pretrained/TNN/doublebell-X1-X8.pth \
  --out-dir viz/doublebell/ \
  --mode pred --bbox-score 0.1
