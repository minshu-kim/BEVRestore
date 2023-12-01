SCALE=$1
GPU=$2

if [ $SCALE == "1" ] ; then
  MPP=0_5
else
  MPP=$(($SCALE / 2))
fi

#CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/test.py \
# configs/nuscenes/seg/TNN-X$SCALE.yaml \
# runs/X$SCALE/latest.pth \
# --eval map

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/test.py \
 configs/nuscenes/seg/TNN-X$SCALE.yaml \
 pretrained/TNN/X$SCALE.pth \
 --eval map
# --night true
# --rainy true
