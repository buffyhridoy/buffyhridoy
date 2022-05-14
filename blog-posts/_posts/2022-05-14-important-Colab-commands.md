---
layout: post
title: Important Colab Commands
description: >
  This blog is a collection of some important colab commands I need to use for training ML or DL models.

canonical_url: http://Important Colab Commands
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---
For connecting Google Colab with Google Drive we need to run the 2 lines of code below:

```pyhon
from google.colab import drive
drive.mount('/content/gdrive')
```
After running these lines of code our Google Drive will be connected to our Colab notebook. Then we can access the data from our Google Drive.
