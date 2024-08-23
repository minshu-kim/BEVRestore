SCALE=$1
GPU=$2
NUM_GPU=$(echo $GPU | tr ',' ' ' | wc -w)

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np $NUM_GPU python tools/train.py \
 configs/nuscenes/seg/bevfusion-bevrestore-X$SCALE-pretrain.yaml \
 --run_dir runs/bevfusion-bevrestore-X$SCALE-pretrain \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --scale $(($SCALE))

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np $NUM_GPU python tools/train.py \
 configs/nuscenes/seg/bevfusion-bevrestore-X$SCALE.yaml \
 --run_dir runs/bevfusion-bevrestore-X$SCALE \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --load_from runs/bevfusion-bevrestore-X$SCALE-pretrain/latest.pth
