# 🖼️ Computer Vision Image Augmentation Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain: Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-blueviolet.svg)](#)
[![Pascal VOC](https://img.shields.io/badge/Format-Pascal%20VOC%20XML-green.svg)](#)

Python utility for augmenting images and automatically transforming bounding box annotations in Pascal VOC XML format.

## 📌 Overview
Data augmentation is essential for training robust object detection and segmentation models. This toolkit provides synchronized image and bounding box transformations—ensuring that whenever an image is flipped, rotated, cropped, or scaled, the corresponding XML annotation bounding box coordinates are precisely recalculated.

## 🛠️ Supported Transformations
- **Spatial:** Horizontal & Vertical Flips, Rotations, Scaled Cropping, Aspect Ratio Jitter.
- **Photometric:** Contrast adjustment, Brightness shifts, Gaussian blur, Channel noise.
- **Annotation Sync:** Automated recalculation of `[xmin, ymin, xmax, ymax]` Pascal VOC XML tags.

## 📂 Architecture
- `img_aug.py`: Main transformation orchestration engine.
- `aug_utils.py`: Coordinate mapping and trigonometric bounding box recalculation.
- `xml_utils.py`: Pascal VOC XML parser, node editor, and exporter.

## 🚀 Quickstart
```bash
git clone https://github.com/FirdowsRahaman/Image-Augmentation.git
cd Image-Augmentation
pip install opencv-python numpy matplotlib
python img_aug.py
```

## 📄 License
This project is licensed under the MIT License.
