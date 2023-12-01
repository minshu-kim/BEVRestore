CONFIG=$1
PRETRAINED=$2
CUDA_VISIBLE_DEVICES=2,3 torchpack dist-run -np 2 python tools/test.py \
 $CONFIG $PRETRAINED --eval bbox
