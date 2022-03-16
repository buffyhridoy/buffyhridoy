---
layout: post
title: Automatic Segmentation and Shape, Texture-based Analysis of Glioma Using Fully Convolutional Network 
image: /assets/img/mini-projects/brain_tumor.jpg
caption: Flow diagram of the experiment
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

## Abstract 

Lower-grade glioma is a type of brain tumor 
that is usually found in the human brain and spinal cord. 
Early detection and accurate diagnosis of lower-grade glioma 
can reduce the fatal risk of the affected patients. An essential 
step for lower-grade glioma analysis is MRI Image 
Segmentation. Manual segmentation processes are time-consuming and depend on the expertise of the pathologist. In 
this study, three different deep-learning-based automatic 
segmentation models were used to segment the tumor-affected 
region from the MRI slice. The segmentation accuracy of the 
three models- U-Net, FCN, and U-Net with ResNeXt50 
backbone were respectively 80%, 84%, and 91%. Two shape-based features- (angular standard deviation, marginal 
fluctuation) and six texture-based features (entropy, local 
binary pattern, homogeneity, contrast, correlation, energy) 
were extracted from the segmented images to find the 
association with seven existing genomic data types. It was 
found out that there was a significant association between the 
genomic data type- microRNA cluster and texture-based 
feature- entropy case and genomic data type- RNA sequence 
cluster with shape-based feature- angular standard deviation 
case. In both of these cases, the p values were observed less 
than 0.05 for the Fisher exact test. 
