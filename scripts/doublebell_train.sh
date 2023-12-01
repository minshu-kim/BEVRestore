GPU=$1


#CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/train.py \
# configs/nuscenes/seg/doublebell.yaml \
# --run_dir runs/scratch/doublebell \
# --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth


CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/train.py \
 configs/nuscenes/seg/doublebell_finetune.yaml \
 --run_dir runs/doublebell_finetune \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --load_from runs/scratch/doublebell/latest.pth
