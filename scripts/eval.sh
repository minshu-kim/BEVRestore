SCALE=$1
GPU=$2
NUM_GPU=$(echo $GPU | tr ',' ' ' | wc -w)

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np $NUM_GPU python tools/test.py \
 configs/nuscenes/seg/bevfusion-bevrestore-X$SCALE.yaml \
 runs/bevfusion-bevrestore-X$SCALE/latest.pth \
 --eval map

