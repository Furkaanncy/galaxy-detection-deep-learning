
# Deep Learning-Based Object Detection in Astronomical Images

## Overview

This repository contains the source codes developed for the
master's thesis titled:

"Astronomical Object Recognition in Astronomical Observations
Using Deep Learning"

The study investigates and compares three deep learning-based
object detection architectures for identifying galaxies in
astronomical images.

## Models

### 1. YOLOv8
- Object detection using YOLOv8.
- Training, validation and prediction procedures.
- Performance evaluation using mAP, precision and recall.

### 2. SSD (Single Shot MultiBox Detector)
- SSD-based object detection.
- Model training and evaluation.
- Detection performance analysis.

### 3. Faster R-CNN (FRCNN)
- Two-stage object detection architecture.
- Feature Pyramid Network (FPN) integration.
- Training and performance evaluation.

## Dataset

The models were developed using an astronomical image
dataset containing two object classes:

- MG: Main Galaxy
- CG: Companion Galaxy

Dataset images and annotations are not included in this
repository.

## Evaluation Metrics

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95
- Confusion Matrix

## Technologies

- Python
- PyTorch
- Torchvision
- Ultralytics
- OpenCV
- NumPy
- Scikit-learn

## Author

Furkan Çağrı Yılmaz

MSc in Space Sciences and Technologies

Çanakkale Onsekiz Mart University
