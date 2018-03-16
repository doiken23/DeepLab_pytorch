# DeepLab family

## Papers

* DeepLab v2 [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
* DeepLab v3 [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* DeepLab v3+ [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

## Contents

* DeepLab v2 (VGG, ResNet101)
* DeepLab v3 (ResNet101)
* DeepLab v3+ (coming soon...)

(DeepLab v2 (VGG16) is a little different from original implementation!!) 

## description

Network | description
:-- | :--
DeepLab v2 (VGG, FOV)| VGG16 + atrous convolution
DeepLab v2 (VGG, ASPP) | VGG16 + atrous spatial pyramid pooling
DeepLab v2 (ResNet, FOV)| ResNet101 + atrous convolution
DeepLab v2 (ResNet, ASPP) | ResNet101 + atrous spatial pyramid pooling
DeepLab v3 | ResNet101 + atrous convolution in cascadea and in parallel
DeepLab v3+ | DeepLab v3 + good decoder (and Xception)
