# portrait segmentation
Segmentation model training pipeline. 

For training, the [SINet model](https://arxiv.org/pdf/1911.09099.pdf) is used with the following changes:
1. SE block and replaced with a better ECA block (dense layer replaced with 1-d conv layer).
2. Replaced the Upsample layer with the TransposedConv layer (with initialization and freezing weights of this layer to the same weights as the Upsample layer) - due to the model compatibility issue with iOS 13.

The pipeline is based on pytorch-lightning with a hydra based config file.