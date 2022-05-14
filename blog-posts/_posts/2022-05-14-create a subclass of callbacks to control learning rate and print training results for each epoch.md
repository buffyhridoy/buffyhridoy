---
layout: post
title: Create a Subclass of Callbacks to Control Learning Rate and Print Training Results for Each Epoch in a Different way
description: >
  In this blog, we have trained a model for showing the subclass of callbacks we have created to control learning rate and print training results for each epoch in a different way
canonical_url: http://Create a Subclass of Callbacks to Control Learning Rate and Print Training Results for Each Epoch in a Different way
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

For this we have used a malaria [dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/) which contain 2 classes.
We have done this whole project on a kaggle notebook.

## Import all dependencies

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
```
