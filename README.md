# BEVRestore

<h3>
  <b> Addressing Diverging Training Costs using BEVRestore <br>
    for High-resolution Bird's Eye View Map Construction </b>
</h3>
[<b><a href="https://arxiv.org/abs/2405.01016">Arxiv</a></b>] [<b>Project Page (TBD)</b>]

## What is BEVRestore? 

  - A novel mechanism containing our introduced bird's eye view (BEV) super-resolution.
  - The method is a plug-and-play mechanism and applicable to any BEV map construction.
  - BEVRestore provides cost efficiency with fine-grained encoding of high-resolution (HR) BEV representation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/39c5e3ca-5ad2-45be-989f-4af38a6347b9" width="300">
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/65441c60-d0b9-43f6-b0c9-00f496d2821e" width="470">
</p>

## News
- [24-09-18] The code is released on this repository. (Setup details will be added.)
- [24-09-18] BEVRestore is accepted at Robotics and Automation Letters (RA-L).
- [24-05-02] BEVRestore is released on [arXiv](https://arxiv.org/abs/2405.01016).

## Model Zoo

We measured training memory containing tensor cache and inference time latency with a single batch.

### 1. BEV Segmentation

- w/ BEVFusion (this repository):

| model        |  BEVRestore |   Latency  |  Memory   |  mIoU    |     weight    |
| ------------ | :---------: | :--------: |  :------: |  :-----: | ------------- |
| BEVFusion    |      -      |     176    |    **3.6**    |    62.7  | [model](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1) |
| BEVFusion    |     X2      |     187    |    4.2    |    69.6  | [LR](https://drive.google.com/file/d/107iSy9yvh-trL2euPHJjFX1__S2bnlq0/view?usp=sharing) / [HR](https://drive.google.com/file/d/1F5eGDX4TGAZnMOTysBSATPig7yiwZCrO/view?usp=sharing) |
| BEVFusion    |     X4      |     164    |    3.8    | **70.9** | [LR](https://drive.google.com/file/d/1aGRS6rm_pklMoXWvMiB_kLGNBdq0RSK9/view?usp=sharing) / [HR](https://drive.google.com/file/d/1McBKHPidOFnu1HVGV-XDA1OAS-3DcQYp/view?usp=sharing) |
| BEVFusion    |     X8      |     **153**    |    3.7    |   70.7   | [LR](https://drive.google.com/file/d/1fYMALBXqDVcFnQDtCMe6_8fwj0zvVMbE/view?usp=sharing) / [HR](https://drive.google.com/file/d/1otVBhGnXBukCDZhkFlbpDvkOhUb0cm5I/view?usp=sharing) |

- w/ BEVFormer ([repository](https://github.com/minshu-kim/BEVRestore-BEVFormer-seg/tree/main)):

|  model                      | X4 BEVRestore | Latency | Memory   | mIoU       |weight |
| --------------------------- | :-------------: | :-------------: | :----------: | :----------: |---------- |
| BEVFomer-small  |           | 125        | 9.9GB      | 40.5     |[model](https://drive.google.com/file/d/1Fn9ErCrWheNFfnCK3EZ1VmCPceUTxS5G/view?usp=share_link)|
| BEVFomer-small |     V      | **104**         | **2.5GB**     | **41.3**     |[LR](https://drive.google.com/file/d/1tUYpqrN6qXYd9uqS2PVViXG7i2GAXpwN/view?usp=sharing) / [HR](https://drive.google.com/file/d/1pNBIRXZl1ZbhupdWAfTGngbOnV_4qIn7/view?usp=share_link)|

### 2. HD Map Construction

- w/ HDMapNet ([repository](https://github.com/minshu-kim/BEVRestore-HDMapNet)):

| model                      | X4 BEVRestore | Latency | Memory   | mAP | mIoU       |weight |
| -------------------------- | :-------------: | :-------------: | :----------: | :----------: | :----------: |---------- |
| HDMapNet |            | **38ms**         | **1.9GB**      | 23.0 | 34.2     |[model](https://drive.google.com/file/d/14UacCCDadgA3L2BRUu62c4XuV7ppXYBd/view?usp=sharing)|
| HDMapNet |     V      | 58ms         | 3.5GB | **32.1**     | **36.4**     |[LR](https://drive.google.com/file/d/1WsIigx9nylSms0KDhUdqejNGtxdjRqFs/view?usp=sharing) / [HR](https://drive.google.com/file/d/1n5rPKPsmYgXhiRIGrCVZSvhVPP-FnI3Z/view?usp=sharing)|

Note: For some vanilla model cases, restoration costs are larger than BEV encoding memory as in this case.
