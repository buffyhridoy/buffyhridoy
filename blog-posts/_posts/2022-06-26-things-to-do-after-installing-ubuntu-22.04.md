---
layout: post
title: Things to Do after Installing Ubuntu 22.04
description: >
  This blog is about things to do after installing ubuntu 22.04.

canonical_url: http://Things to Do after Installing Ubuntu 22.04
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

Follow these quick tips to do after installing Ubuntu 22.04

### 1. Check and Install Package Updates
The first step is to check and install updates to keep our computer’s software up to date. This is the single most important task we need to do to protect our system.
To install updates, open a terminal window and simply run the following commands.

```python
sudo apt-get update && sudo apt-get dist-upgrade
```
### 2. Install VLC Media Player
VLC is a simple yet powerful and widely-used multimedia player and framework that plays most if not all multimedia files. It also plays DVDs, Audio CDs, VCDs as well as numerous streaming protocols.

It is distributed as a snapcraft for Ubuntu and many other Linux distributions. To install it, open a terminal window and run the following command.

```python
sudo snap install vlc
```
### 3. Install Media Codecs
The Ubuntu maintainers want to include only free and open-source software, closed-source packages such as media codecs for common audio and video files such as MP3, AVI, MPEG4, and so on, are not provided by default in a standard installation. To install them, we need to install the ubuntu-restricted-extras meta-package by running the following command.

```python
sudo apt install ubuntu-restricted-extras
```
### 4. Install GNOME Tweaks
GNOME Tweaks is a simple graphical interface for advanced GNOME 3 settings. It enables us to easily customize our desktop. Although it is designed for the GNOME Shell, we can use it in other desktops.

```python
sudo apt install gnome-tweaks
```
### 5. Select Default Applications
In any desktop operating system, once you double-click a file in the file manager, it will be opened with the default application for that file type. To configure the default applications to open a file type in Ubuntu 22.04, go to Settings, then click Default Applications, and select them from the drop-down menu for each category.

![image](https://user-images.githubusercontent.com/37147511/175800893-c08fba72-ebff-4e7d-8b36-f008b77b1a78.png)

### 6. Configure Keyboard Shortcuts
Using keyboard shortcuts can increase your productivity and save you lots of time when using a computer. To set our keyboard shortcuts, under Settings, simply click on Keyboard Shortcuts.

![image](https://user-images.githubusercontent.com/37147511/175800942-e49851b2-667d-4a9d-8092-86ae0429e3af.png)

### 7. Install Wine for Running Windows Apps
If we intend to run Windows applications in Ubuntu 22.04, then we need to install Wine – is an open-source implementation of the Windows API on top of X and POSIX-compliant operating systems, such as Linux, BSD, and macOS. It allows us to integrate and run Windows application cleanly, on Linux desktops by translating Windows API calls into POSIX calls on-the-fly. To install Wine, run this command.

```python
sudo apt install wine winetricks
```
### 8. Add Favorite Apps to the Dock
To add our favorite applications to the Ubuntu Dock (which is situated on the left side of your desktop by default), click on the Activities overview, search for the application we want e.g terminal, then right-click on it and select Add to Favorites.

![image](https://user-images.githubusercontent.com/37147511/175801024-9221a617-135e-40df-b15b-b1b327b038d0.png)

### 9. Install Laptop Power Saving Tools
If we are using a laptop, then we might want to install Laptop Mode Tools, a simple and configurable laptop power-saving tool for Linux systems. It helps to extend our laptop’s battery life in so many ways. It also allows us to tweak some other power-related settings using a configuration file.

```python
sudo apt install laptop-mode-tools
```
