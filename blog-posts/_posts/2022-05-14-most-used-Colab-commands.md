---
layout: post
title: Most Used Colab Commands
description: >
  This blog is a collection of some important colab commands I need to use for training ML or DL models.

canonical_url: http://Most Used Colab Commands
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---
## Connect Google Colab with Google Drive 

For connecting Google Colab with Google Drive we need to run the 2 lines of code below:

```pyhon
from google.colab import drive
drive.mount('/content/gdrive')
```
After running these lines of code our Google Drive will be connected to our Colab notebook. Then we can access the data from our Google Drive.

## Accessing local file system from Colab notebook

First we need to import the `files` module from the google.colab library. Then we can use the `upload` method of the files object

```pyhon
from google.colab import files
uploaded = files.upload()
```
After running this block of code we will see following interface for uploading the files from local system.

![image](https://user-images.githubusercontent.com/37147511/168416437-325ed998-a878-4321-b132-9b526a86da37.png)

