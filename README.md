# Road Defect Detection in EdgeAi (2025)

By Matti, Elias and Jaakko

## About

Training YOLOv8m on the [RDD2022 dataset](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) to detect road defects, and then quantizing it and optimizing it to run on a RaspberryPi + TPU.

## Dependencies

There are two main dependencies, PyTorch and ultralytics (for YOLO, which requires PyTorch). It is best to install PyTorch first with a separate command and then install the requirements in requirements.txt.

### Install steps:

1. Make sure pip is updated (For Windows `python.exe -m pip install --upgrade pip`)
2. Install PyTorch (see https://pytorch.org/get-started/locally/)
3. Install requirements.txt (`pip install -r requirements.txt`)

HUOM: PyTorch is not listed in requirements.txt, but the version used is `2.6.0+cul26`.
