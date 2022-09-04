---
layout: post
title: Installing Tensorflow-GPU
description: >
  This blog is about installing tensorflow-gpu in local machine.

canonical_url: http://Installing Tensorflow-GPU
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---


## 1st Step
For installing CUDA and cuDNN with conda.

```pyhon
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## 2nd Step
For installing tensorflow-gpu

```pyhon
pip install tensorflow-gpu
```

