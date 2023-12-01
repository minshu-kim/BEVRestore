CUDA_VISIBLE_DEVICES=$1 torchpack dist-run -np 2 python tools/train.py \
 configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevsr.yaml \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --load_from pretrained/bevfusion-det.pth
