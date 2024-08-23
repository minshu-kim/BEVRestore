# BEVRestore (2024 RA-L)

![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/65441c60-d0b9-43f6-b0c9-00f496d2821e)


# Model

We measured training memory containing tensor cache and inference time latency with a single batch.

| model        |  BEVRestore |   Latency  |  Memory   |  mIoU    |     weight    |
| ------------ | :---------: | :--------: |  :------: |  :-----: | ------------- |
| BEVFusion    |      -      |     176    |    3.6    |    62.7  | [model](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1) |
| BEVFusion    |     X2      |     187    |    4.2    |    69.6  | [LR](https://drive.google.com/file/d/107iSy9yvh-trL2euPHJjFX1__S2bnlq0/view?usp=sharing) / [HR](https://drive.google.com/file/d/1F5eGDX4TGAZnMOTysBSATPig7yiwZCrO/view?usp=sharing) |
| BEVFusion    |     X4      |     164    |    3.8    | **70.9** | [LR](https://drive.google.com/file/d/1aGRS6rm_pklMoXWvMiB_kLGNBdq0RSK9/view?usp=sharing) / [HR](https://drive.google.com/file/d/1McBKHPidOFnu1HVGV-XDA1OAS-3DcQYp/view?usp=sharing) |
| BEVFusion    |     X8      |     153    |    3.7    |   70.7   | [LR](https://drive.google.com/file/d/1fYMALBXqDVcFnQDtCMe6_8fwj0zvVMbE/view?usp=sharing) / [HR](https://drive.google.com/file/d/1otVBhGnXBukCDZhkFlbpDvkOhUb0cm5I/view?usp=sharing) |
