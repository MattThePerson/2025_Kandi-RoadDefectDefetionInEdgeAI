# Road Defect Detection for EdgeAI (2025)

By Matt Stirling, Elias Kyyhkynen and Jaakko Koivuvaara.

## About

Training [YOLOv8n](https://docs.ultralytics.com/models/yolov8/#overview) on the [RDD2022 dataset](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) to detect road defects, and then quantizing it and optimizing it to run on a RaspberryPi + TPU.

## Dependencies

There are two main dependencies, PyTorch and ultralytics (for YOLO, which requires PyTorch). It is best to install PyTorch first with a separate command and then install the requirements in requirements.txt.


## Install

PyTorch recommends using PyPi over Anaconda. Please do one of the following options:

### **Option 1:** venv + pip

1. Create venv: `python -m venv .venv`
2. Activate venv \
    Windows: `.venv\Scripts\activate` \
    Linux: `source .venv/bin/activate`
3. Install PyTorch (see https://pytorch.org/get-started/locally/) \
    `pip install torch ...` (please see the above link)
4. Install requirements.txt \
    `pip install -r requirements.txt`


### **Option 2:** uv (recommended)

uv is a fast, modern package/project manager for Python. Can be [installed](https://docs.astral.sh/uv/getting-started/installation/) for both Windows and Linux. 

1. Create venv: `uv venv`
2. *No need to activate venv seperately*
3. Install PyTorch (use `uv pip install` instead of just `pip install`)
4. Install requirements.txt \
    `uv pip install -r requirements.txt`


## Running

Suppose you want to run the script `scripts/preprocessing.py`.

1. Make sure you're at the project root

### venv + pip

1. Activate venv \
    Windows: `.venv\Scripts\activate` \
    Linux: `source .venv/bin/activate`
2. Run script: \
    `python scripts/preprocessing.py`


### uv

No need to activate venv separately. Just run: 

`uv run scripts/preprocessing.py`

> [!IMPORTANT]  
> These instructions assume 

## Additional Notes

HUOM: PyTorch is not listed in requirements.txt, but the version used is `2.6.0+cul26`.


> [!TIP]  
> Don't worry about get

 [!WARNING]  
> Snap-based Do

```sh
sudo docker info | grep "Docker Root Dir" | grep "/var/snap/docker/"
```

Text

```python
def main():
    print("Hello world!")
```