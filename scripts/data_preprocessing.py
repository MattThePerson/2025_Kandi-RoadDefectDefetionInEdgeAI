# Matt Stirling  2025-01-31
# Script for preprocessing RDD2022 dataset to be compatible with YOLO
# RDD2022 dataset: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547
# 
# Things this will change:
# 1. Splits `train` folder into train, val, test folders
#    - saves into `images/<TYPE>/` folder
# 2. Reads image data (including labels) from .xml file
#    - file location: `train/annotations/xmls/<FILENAME>.xml`
# 3. Saves labels in YOLOs preferred format:
#    - bbox normalized -> [0,1]
#    - saves to `labels/<TYPE>/<FILENAME>.txt`
# 
# HUOM: `test` folder doesnt contain labels, generates test data from `train` folder
from typing import Any
import argparse
from pathlib import Path
import os
import time
import random
import xml.etree.ElementTree as ET
import shutil

random.seed(0)


def main(args: argparse.Namespace):
    dataset_unprocessed = r'..\datasets\RDD2022'
    dataset_processed =   r'..\datasets\RDD2022_preprocessed' # location of processed dataset

    subfolders = list(Path(dataset_unprocessed).glob('*'))

    print('Preprocessing RDD2022 dataset into folder: "{}"'.format(dataset_processed))

    handled_count = 0
    start = time.time()
    for idx, subfolder in enumerate(subfolders):
        print('  ({}/{}) Handling subfolder: "{}"'.format(idx+1, len(subfolders), subfolder.name))
        folderpath = Path(dataset_processed)# / subfolder.name
        images_handled = preprocess_subfolder(
            subfolder,
            folderpath,
            portion=args.fraction,
            copy_files=(not args.move_files),
            test_run=args.test_run,
        )
        handled_count += images_handled
        time.sleep(0.112)

    print('\nDone. Pre-processed {:_} images in {:.1f}s'.format(handled_count, (time.time()-start)))
    print('Preprocessed data folder: "{}"'.format(str(dataset_processed)))
    
    return

def preprocess_subfolder(
    subfolder: Path,
    target: Path,
    copy_files: bool = True, # if false, moves instead of copy
    train_val_test_split: Any = (80,10,10),
    portion: float = 1.0, # portion (0 to 1) of dataset to copy/move and preprocess
    test_run: bool = False,
):
    """ Preprocess the data in a given subfolder in the RDD2022 dataset """
    # read all images in 'train' folder ()
    images_folder = subfolder / Path('train/images')
    images = list(images_folder.glob('*'))
    print('Found {:_} images'.format(len(images)))
    if portion < 1.0:
        images = images[:int(len(images)*portion)]
        print('Limiting images to {:_} ({:.0f}%)'.format(len(images), portion*100))
    
    # splits images into train-val-test by given split
    train_imgs, val_imgs, test_imgs = split_array(images, train_val_test_split)
    
    # moves/copies images into target folder and 
    dataset_split = { 'train': train_imgs, 'val': val_imgs, 'test': test_imgs }
    for type, imgs in dataset_split.items():
        images_dirname = os.path.join( str(target), 'images', type )
        labels_dirname = os.path.join( str(target), 'labels', type )
        os.makedirs(images_dirname, exist_ok=True)
        os.makedirs(labels_dirname, exist_ok=True)
        for idx, img in enumerate(imgs):
            print('\rProcessing \'{}\' image ({}/{}): "{}"     '.format(type, idx+1, len(imgs), img.name), end='')
            xml_path = get_image_xml_path(img)
            labels_path = os.path.join( labels_dirname, f'{img.stem}.txt' )
            labels_str = xml_file_to_yolo_label_string(xml_path)
            if not test_run:
                with open(labels_path, 'w') as f:
                    f.write(labels_str)
                copy_file(str(img), images_dirname, copy_file=copy_files)
        print()
    
    return len(images)


# region helpers

def split_array(items: list[Path], split: Any) -> tuple[list[Path], list[Path], list[Path]]:
    items = items.copy()
    random.shuffle(items)
    split1_idx = int( len(items) * split[0]/100 )
    split2_idx = int( len(items) * (split[0] + split[1])/100 )
    train = items[:split1_idx]
    val = items[split1_idx:split2_idx]
    test = items[split2_idx:]
    return (train, val, test)

def get_image_xml_path(imagepath: Path) -> str:
    base_dir = imagepath.parent.parent
    xml_path = base_dir / Path(f'annotations/xmls/{imagepath.stem}.xml')
    return str(xml_path)


label_map = {
    "D00" :  0,
    "D01" :  1,
    "D0w0" :  2,
    "D10" :  3,
    "D11" :  4,
    "D20" :  5,
    "D40" :  6,
    "D43" :  7,
    "D44" :  8,
    "D50" :  9,
    "Repair" :  10,
    "Block crack" :  11,
}


def xml_file_to_yolo_label_string(xml_filename: str) -> str:
    global label_map
    try:
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        size = root.find('size')
        wid, hei = float(size.find('width').text), float(size.find('height').text) # type: ignore
        detection_strings: list[str] = []
        for object in root.findall('object'):
            name = object.find('name').text # type: ignore
            id_ = label_map.get(name) # type: ignore
            if id_ == None:
                raise Exception('Label name without id: "{}"'.format(name))
            bbox = object.find('bndbox')
            sizes = []
            for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                sizes.append(float(bbox.find(tag).text)) # type: ignore
            x_center, y_center, width, height = format_bndbox_to_yolo(wid, hei, *sizes) # type: ignore
            det = f'{id_} {x_center} {y_center} {width} {height}'
            detection_strings.append(det)
        return '\n'.join(detection_strings)
    except Exception as e:
        print('\n\nERROR Parsing XML:', e)
        print('content:\n')
        with open(xml_filename, 'r') as f:
            for lin in f:
                print(lin, end='')
        print('\n\nEXITING')
        exit(1)


def format_bndbox_to_yolo(img_width: float, img_height: float, xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """ Converts bounding boxes to format: x_center y_center width height """
    width = xmax - xmin
    x_center = xmin + width / 2
    x_center, width = x_center/img_width, width/img_width
    
    height = ymax - ymin
    y_center = ymin + height / 2
    y_center, height = y_center/img_height, height/img_height
    
    return x_center, y_center, width, height


def copy_file(src: str, dst: str, copy_file:bool = True) -> bool:
    try:
        if copy_file:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        return True
    except:
        return False


# endreion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test-run', action='store_true', help='Dont move images or create labels files (still creates directories)')
    parser.add_argument('--fraction', type=float, default=1.0, help='The fraction of the dataset to pre-process')
    parser.add_argument('--move-files', action='store_true', help='Moves images instead of copying images')

    args = parser.parse_args()

    print()
    try:
        main(args)
    except KeyboardInterrupt:
        print('\n\n... halting preprocessing')
    print()
