SCALE=$1
GPU=$2

if [ $SCALE == "1" ] ; then
  MPP=0_5
  mpp=0.5

else
  MPP=$(($SCALE / 2))
  mpp=MPP
fi

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/train.py \
 configs/nuscenes/seg/$MPP-mpp.yaml \
 --run_dir runs/scratch/$MPP-mpp \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --scale $(($SCALE))

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/train.py \
 configs/nuscenes/seg/TNN-X$SCALE.yaml \
 --run_dir runs/X$SCALE \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --load_from runs/scratch/$MPP-mpp/latest.pth
# --resume_from runs/X$SCALE/latest.pth
