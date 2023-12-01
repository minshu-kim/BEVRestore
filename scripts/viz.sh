SCALE=$1

CUDA_VISIBLE_DEVICES=3 torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/seg/TNN-X$SCALE.yaml \
  --checkpoint pretrained/TNN/X$SCALE.pth \
  --out-dir viz/TNN-X$SCALE/ \
  --mode pred --bbox-score 0.1

#  --mode pred --bbox-score 0.1 --doublebell false
