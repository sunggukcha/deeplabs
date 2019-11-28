# Deeplabv3+ BDD100k/drivable_area implementation

![Results](./images/result.png)

### Introduction

This repository is merged with https://github.com/jfzhang95/pytorch-deeplab-xception and is success of https://github.com/sunggukcha/deeplabv3plus-bdd100k-drivablearea. Please follow installation policies of the repositories above. 

For BDD100k/drivable_area semantic segmentation, I added 

1. bdd100k drivable area dataloader, and training/val/test scripts.
2. prediction visualization for both color (visual result) and id (greyscale png file for submission).
3. added Group Noramlization.
4. deeplabv3 which is without deeplabv3+ decoder, but with aspp only. 
5. WRN as backbone is added (original code from mapillary@github)
6. additional visualization that marks corrects, missed and wrong pixels.
7. IBN-Net by github.com/XingangPan/IBN-Net/
8. EfficientNet added which is implemented by https://github.com/lukemelas/EfficientNet-PyTorch.
9. Feature Pyramid Networks(FPNs) for semantic segmentation added (version: Panoptic Feature Pyramid Networks).


For more detail, please visit the repositories above.

### Experiment & result
  
 <span style="color:red">**Single 12GB GPU**</span>

| Backbone  | Normalization  |mIoU in test |
| :-------- | :------------: |:-----------:|
| ResNet50  | Group-16       | 85.00%      |
| ResNet101 | IGN-a-16       | 85.12%      |
| ResNet101 | Group-16       | 85.33%      |
| ResNet152 | Group-16       | 85.45%      |

IGN-a-16 denotes instance group normalization with channel-grouping number 16, replacing BN of IBNNet-a with GN16.
Group-16 denotes group normalization with channel-grouping number 16.

| WAD2018   | Score          | Difference   |
| :-------- | :------------: |:-----------: | 
| 1st       | 86.18          | -0.09 |
| Mine      | 86.09          | +0.0  |
| 2nd       | 86.04          | +0.05 |
| 3rd       | 84.01          | +2.08 |

