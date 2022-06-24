---
layout: post
title: Publish a Jupyter Notebook as a Blog Post on Github Pages
description: >
  This blog is about publishing jupyter notebook as a blog post on github pages.

canonical_url: http://Publish a Jupyter Notebook as a Blog Post on Github Pages
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

We more and more work in Jupyter Notebook. How can we show our notebook as a blog post on github pages? 
There is an easy command to convert our .ipynb files to .md files, namely _**nbconvert**_. 
We have to install a library using following command:
```python
pip install nbconvert
```
We can do this easily in our terminal. But first, we have to change the directory we are working in to the one of our notebook.ipynb file.
Then we can transform our jupyter notebook through a simple command in the terminal:

```python
jupyter nbconvert --to markdown  notebook.ipynb 
```
This then creates a notebook.md file with the markdown text for our blog post.

Sometimes we might encounter the problem that not all of our notebook converts but we have to install another library by following command:
```python
pip install pdflatex
```
Then we have to try the following in our teminal:

```python
Jupyter nbconvert --to latex notebook.ipynb
pdflatex notebook.tex
```
