# Road Defect Detection for EdgeAI (2025)

By Matt Stirling, Elias Kyyhkynen and Jaakko Koivuvaara.

## About

Training [YOLOv8n](https://docs.ultralytics.com/models/yolov8/#overview) on the [RDD2022 dataset](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) to detect road defects, and then quantizing it and optimizing it to run on a RaspberryPi + Edge TPU.

## Dependencies

There are two main dependencies, PyTorch and ultralytics (for YOLO, which requires PyTorch). It is best to install PyTorch first with a separate command and then install the requirements in requirements.txt.


## Install

PyTorch recommends using PyPi over Anaconda. Please do one of the following options:

### **Option 1:** venv + pip

1. Create venv: `python -m venv .venv`
2. Activate venv    
    Windows: `.\venv\Scripts\activate.bat`  
    Linux: `source .venv/bin/activate`
3. Install [PyTorch](https://pytorch.org/get-started/locally/)  
    `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
4. Install requirements.txt \
    `pip install -r requirements.txt`


### **Option 2:** uv (recommended)

uv is a fast, modern package/project manager for Python. Can be [installed](https://docs.astral.sh/uv/getting-started/installation/) for both Windows and Linux. 

1. Create venv: `uv venv`
2. *No need to activate venv seperately*
3. Install [PyTorch](https://pytorch.org/get-started/locally/)  
    `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
4. Install requirements.txt     
    `uv pip install -r requirements.txt`


## Running

Suppose you want to run the script `scripts/preprocessing.py`. Make sure you're at the project root, then:

### **Option 1:** venv + pip

1. Activate venv    
    Windows: `.\venv\Scripts\activate.bat`  
    Linux: `source .venv/bin/activate`
2. Run script:  
    `python scripts/preprocessing.py`


### **Option 2:** uv

run: `uv run scripts/preprocessing.py`


### (Optional) Verify python executable path

You should make sure python points to the local python interpreter. Make sure the python path points to the local `.venv` folder.

1. With activated venv:     
    Windows: `Get-Command python`   
    Linux: `which python`
2. With uv  
    `uv python find`


### (Optional) Verify PyTorch install

To verify PyTorch works, and that CUDA is available (if installed CUDA version), you can create and run a demo python script

```python
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
```
