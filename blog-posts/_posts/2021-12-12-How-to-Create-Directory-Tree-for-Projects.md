---
layout: post
title: How to Create Directory Tree for Projects
excerpt: A simple way of creating Directory Tree for any projects 

---

First we have to install a python package called directory-tree-generator. Installing process has been shown in the image below.  

![Screenshot (238)](https://user-images.githubusercontent.com/37147511/145712326-43bc535a-3181-4dba-9a8a-382ed3ec708b.png)

After installing this, we have to write just 3 lines of code for generating directory tree

```python
from DirectoryTree import TreeGenerator
Tree = TreeGenerator()
Tree.generate('D:\Retinal Inference Project')
```

In the line 3,  `Tree.generate('D:\Retinal Inference Project')` we have passed the root directory of out project.
Here is the of output which showing the directory tree of our project

```python
+ Retinal Inference Project/
    • blend2.png
    • data_aug.py
    • demo.ipynb
    • demo1.ipynb
    • inference.py
    • output2.png
    • test.py
    • train.py
   + buffy/
       • loss.py
       • model.py
       • utils.py
       • __init__.py
      + __pycache__/
          • loss.cpython-37.pyc
          • model.cpython-37.pyc
          • __init__.cpython-37.pyc
   + config/
       • weights_download.json
   + data/
       • data.py
       • data_aug.py
   + files/
       • .gitattributes
       • checkpoint.pth
   + img/
       • image.png
       • mask.png
```
