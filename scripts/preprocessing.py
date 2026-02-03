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
import datetime
import json
import sys
import yaml
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
    if args.fraction < 0 or args.fraction > 1.0:
        raise ValueError(f'Fraction {args.fraction} given as argument is not between 0 and 1.0')
    dataset = Path(args.path)
    # print(f'absolute path: {dataset_unprocessed.absolute()}')
    if not dataset.is_dir():
        raise NotADirectoryError(f'{args.path} is not a directory. Double-check the path given as an argument, the path'
                                 f' should be relative: this/is/relative VS /this/is/absolute. The path you gave was: '
                                 f'{dataset.absolute()}')

    # dataset_unprocessed = os.path.join('datasets', 'RDD2022')
    # dataset_processed = os.path.join('datasets', 'RDD2022_YOLO') # location of processed dataset

    handled_count = 0
    start = time.time()

    ### XML ###
    if not args.json:  # data is in xml format

        dataset_processed = dataset.parent / args.name
        if dataset_processed.exists() and not args.yes:
            answer = input(f'A preprocessed dataset with this name already exists: {dataset_processed}\n'
                           f'Do you want to replace the previous preprocessed dataset? Y/n\n')
            if answer != 'Y' and str.lower(answer) != 'yes':
                print('Answer was not "Y" or "yes", exiting the program.')
                sys.exit(1)
            else:
                # create / replace old preprocessed folder
                if dataset_processed.exists():
                    shutil.rmtree(dataset_processed)
                    print(f"Deleted old dataset with the same name: {dataset_processed}")
                os.makedirs(dataset_processed)
        print(f'Preprocessing RDD2022 dataset into folder: "{dataset_processed}"')

        subfolders = list(dataset.glob('*'))
        for idx, subfolder in enumerate(subfolders):
            print('  ({}/{}) Handling subfolder: "{}"'.format(idx + 1, len(subfolders), subfolder.name))
            folderpath = dataset_processed  # / subfolder.name

            images_handled = preprocess_subfolder_xml(
                subfolder,
                folderpath,
                portion=args.fraction,
                copy_files=(not args.move_files),
                test_run=args.test_run,
                image_list=args.image_list
            )
            handled_count += images_handled
            time.sleep(0.112)
        print('Preprocessed data folder: "{}"'.format(str(dataset_processed)))
    ### XML ends ###

    ### JSON ###
    else:  # data is in json format
        dataset_processed = dataset.parent / args.name
        if dataset_processed.exists() and not args.yes:
            answer = input(f'A preprocessed dataset with this name already exists: {dataset_processed}\n'
                           f'Do you want to replace the previous preprocessed dataset? Y/n\n')
            if answer != 'Y' and str.lower(answer) != 'yes':
                print('Answer was not "Y" or "yes", exiting the program.')
                sys.exit(1)
            else:
                # create / replace old preprocessed folder
                if dataset_processed.exists():
                    shutil.rmtree(dataset_processed)
                    print(f"Deleted old dataset with the same name: {dataset_processed}")
                os.makedirs(dataset_processed)

        # subfolder = dataset / 'train'
        images_handled = preprocess_subfolder_json(
            dataset,
            dataset_processed,
            portion=args.fraction,
            copy_files=(not args.move_files),
            test_run=args.test_run,
            filter_terms=args.exclude
        )
        handled_count += images_handled
        time.sleep(0.112)
        print(f'\nYOLO compatible dataset folder: "{str(dataset_processed)}"')

    ### JSON ends ###

    # create a .yaml file
    create_yaml(args)

    print(f'Done. Pre-processed {handled_count} images in {(time.time() - start):.1f}s')

    return


def preprocess_subfolder_xml(
        subfolder: Path,
        target: Path,
        copy_files: bool = True,  # if false, moves instead of copy
        train_val_test_split: Any = (80, 10, 10),
        portion: float = 1.0,  # portion (0 to 1) of dataset to copy/move and preprocess
        test_run: bool = False,
        image_list: Path = None
):
    """ Preprocess the data in a given subfolder in the RDD2022 dataset """
    # read all images in 'train' folder ()
    images_folder = subfolder / Path('train/images')
    images = list(images_folder.glob('*'))
    print('Found {:_} images'.format(len(images)))
    if portion < 1.0:
        images = images[:int(len(images)*portion)]
        print('Limiting images to {:_} ({:.0f}%)'.format(len(images), portion*100))

    # create image split from a predefined list
    if image_list:
        dataset_dict = read_images_from_list(Path(image_list))
        train_imgs = []
        val_imgs = []
        test_imgs = []
        for img in images:
            if dataset_dict[img.name] == 'train':
                train_imgs.append(img)
            elif dataset_dict[img.name] == 'val':
                val_imgs.append(img)
            elif dataset_dict[img.name] == 'test':
                test_imgs.append(img)

        dataset_split = { 'train': train_imgs, 'val': val_imgs, 'test': test_imgs }
    # create image split randomly
    else:
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
                # always create label file
                with open(labels_path, 'w') as f:
                    f.write(labels_str)
                copy_file(str(img), images_dirname, copy_file=copy_files)
        print()

    return len(images)

def read_images_from_list(image_folder: Path):
    """

    @param image_folder: a folder containing three text files: train.txt, test.txt and val.txt containing a list of image names, one image per line
    @return: a dictionary of images, where key = "image_name.jpg" and value is "train", "val", "test"
    """
    def read_file(file):
        with open(image_folder.joinpath(file), "r") as f:
            images = f.read().splitlines()
        f.close()
        return images

    image_dict = dict()
    image_dict.update({image: "train" for image in read_file("train.txt")})
    image_dict.update({image: "val" for image in read_file("val.txt")})
    image_dict.update({image: "test" for image in read_file("test.txt")})
    return image_dict

def preprocess_subfolder_json(
        dataset: Path,
        dataset_processed: Path,
        copy_files: bool = True,  # if false, moves instead of copy
        train_val_test_split: Any = (80, 10, 10),
        portion: float = 1.0,  # portion (0 to 1) of dataset to copy/move and preprocess
        test_run: bool = False,
        filter_terms=None
):
    """
    """
    if filter_terms is None:
        filter_terms = []
    else:
        filter_terms = args.exclude.split(' ')
    filter_terms = set(filter_terms)

    # read all images in 'train' folder ()
    annotations_folder = dataset / 'train' / 'ann'
    annotations = list(annotations_folder.glob('*'))

    print(f"Filter terms: {filter_terms}")
    # remove files containing filter terms
    filtered_annotations = []
    for ann in annotations:
        include = True
        for term in filter_terms:
            if ann.name.find(term) > -1:
                include = False
                break
        if include:
            filtered_annotations.append(ann)
    print(f"Filtered out {len(annotations) - len(filtered_annotations)} images from the dataset.")
    annotations = filtered_annotations

    # glob does not guarantee that the list is always in the same order
    # sort the list of images for reproducibility
    annotations.sort()

    print('Found {:_} images'.format(len(annotations)))
    if portion < 1.0:
        annotations = annotations[:int(len(annotations) * portion)]
        print('Limiting images to {:_} ({:.0f}%)'.format(len(annotations), portion * 100))

    # splits images into train-val-test by given split
    train, val, test = split_array_json(annotations, train_val_test_split)

    # moves/copies images into target folder
    dataset_split = {'train': train, 'val': val, 'test': test}
    for type, anns in dataset_split.items():
        images_dirname = os.path.join(str(dataset_processed), 'images', type)
        labels_dirname = os.path.join(str(dataset_processed), 'labels', type)
        os.makedirs(images_dirname, exist_ok=True)
        os.makedirs(labels_dirname, exist_ok=True)
        for idx, ann in enumerate(anns):
            print('\rProcessing \'{}\' image ({}/{}): "{}"     '.format(type, idx + 1, len(anns), ann.name), end='')
            labels_str = json_file_to_yolo_label_string(ann)
            json_path = os.path.join(dataset, ann.name.replace('.jpg.json', '.txt'))
            labels_path = os.path.join(labels_dirname, ann.name.replace('.jpg.json', '.txt'))
            if not test_run:
                # only create label file if a label exists
                # if labels_str:
                with open(labels_path, 'w') as f:
                    f.write(labels_str)
                    image_source = str(ann).replace('/ann/', '/img/')[:-5]
                copy_file(image_source, images_dirname, copy_file=copy_files)
                # .removesuffix('.json'), images_dirname, copy_file=copy_files)
        print()

    return len(annotations)

xml_label_map = {
    "D00": 0,
    #"D01" :  1,
    #"D0w0" :  2,
    "D10": 1,
    #"D11" :  4,
    "D20": 2,
    "D40": 3,
    #"D43" :  7,
    #"D44" :  8,
    #"D50" :  9,
    #"Repair" :  10,
    #"Block crack" :  11,
}

# TODO: double check that these labels are correct
json_label_map = {
    6511999: 0,  # D00, "wheel mark part", "longitudinal crack"
    #"D01" :  1,
    #"D0w0" :  2,
    6512000: 1,  # D10, "equal interval", "transverse crack"
    #"D11" :  4,
    6512001: 2,  # D20, "partial pavement, overall pavement", "alligator crack"
    6512002: 3,  # D40, "rutting, bump, pothole, separation", "pothole"
    #"D43" :  7,
    #"D44" :  8,
    #"D50" :  9,
    #"Repair" :  10,
    #"Block crack" :  11,
}

yaml_label_map = {
    0: "D00",
    # 1: "D01",
    # 2: "D0w0",
    1: "D10",
    # 4: "D11",
    2: "D20",
    3: "D40",
    # 7: "D43",
    # 8: "D44",
    # 9: "D50",
    # 10: "Repair",
    # 11: "Block crack"
}

def json_file_to_yolo_label_string(file_path: Path) -> str:
    """returns all labels as a string: 'damage_id_number x_center, y_center, width, height \n ...' """
    with open(file_path, 'r') as file:
        data = json.load(file)

    width = data['size']['width']
    height = data['size']['height']
    objects = ""
    labels = []

    for o in data['objects']:
        if o['classId'] in json_label_map:
            labels.append(str(json_label_map[o['classId']]) + " ")
            labels.append(' '.join(map(str, format_bndbox_to_yolo(width, height,
                                                                  o['points']['exterior'][0][0],
                                                                  o['points']['exterior'][0][1],
                                                                  o['points']['exterior'][1][0],
                                                                  o['points']['exterior'][1][1]))))
            labels.append('\n')
    return objects.join(labels)


sorted_images_hash = 0
shuffled_images_hash = 0

def split_array_json(items, split: Any):

    # normalize the values in split
    split = tuple(x / sum(split) for x in split)

    image_ratio = {
        'China_Drone': [0, 0, 0],
        'China_MotorBike': [0, 0, 0],
        'Czech': [0, 0, 0],
        'India': [0, 0, 0],
        'Japan': [0, 0, 0],
        'Norway': [0, 0, 0],
        'United_States': [0, 0, 0]
    }

    train_val_test = ([], [], [])

    items = items.copy()

    # sorted_hash = 5381
    # for i, item in enumerate(items):
    #     sorted_hash = (((sorted_hash << 5) + sorted_hash) + int(item.name[-15:-9]) ) % 2**64
    # print(f"Sorted list hash: {str(sorted_hash)[-16:]}")

    # shuffle items
    random.shuffle(items)

    shuffled_hash = 5381
    for i, item in enumerate(items):
        shuffled_hash = (((shuffled_hash << 5) + shuffled_hash) + int(item.name[-15:-9]) ) % 2**64
    # print(f"Shuffled list hash: {str(shuffled_hash)}")

    # save the hashes for the .yaml file
    # global sorted_images_hash
    # sorted_images_hash = str(sorted_hash)
    global shuffled_images_hash
    shuffled_images_hash = str(shuffled_hash)

    # place items into train, val and test
    for path in items:
        # E.g. 'China_Drone' or 'Norway'
        image_source = path.name.rsplit('_', 1)[0]

        # find the category with the lowest ratio
        ratio = image_ratio.get(image_source)
        image_destination = ratio.index(min(ratio))


        # place the image in that category
        train_val_test[image_destination].append(path)

        # update the ratio
        ratio[image_destination] += 1 / split[image_destination]

    return train_val_test


def split_array(items, split: Any):
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




def xml_file_to_yolo_label_string(xml_filename: str) -> str:
    #global label_map
    try:
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        size = root.find('size')
        wid, hei = float(size.find('width').text), float(size.find('height').text)  # type: ignore
        detection_strings: list[str] = []
        for object in root.findall('object'):
            name = object.find('name').text  # type: ignore
            id_ = xml_label_map.get(name)  # type: ignore

            if name is None:
                raise Exception('Label name without id: "{}"'.format(name))

            # only labels D00, D10, D20, D40 are relevant for our project
            if name in (['D00', 'D10', 'D20', 'D40']):
                bbox = object.find('bndbox')
                sizes = []
                for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                    sizes.append(float(bbox.find(tag).text))  # type: ignore
                x_center, y_center, width, height = format_bndbox_to_yolo(wid, hei, *sizes)  # type: ignore
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


def format_bndbox_to_yolo(img_width: float, img_height: float, xmin: float, ymin: float, xmax: float, ymax: float):
    """ Converts bounding boxes to format: x_center y_center width height """
    width = xmax - xmin
    x_center = xmin + width / 2
    x_center, width = x_center / img_width, width / img_width

    height = ymax - ymin
    y_center = ymin + height / 2
    y_center, height = y_center / img_height, height / img_height

    return x_center, y_center, width, height


def copy_file(src: str, dst: str, copy_file: bool = True) -> bool:
    try:
        if copy_file:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        return True
    except:
        return False


def create_yaml(args: argparse.Namespace):

    filename = args.name + '.yaml'

    # save the .yaml file in the current working directory
    save_location = 'YAML/'

    comments = (f"# This is a YAML file for the dataset {args.name}\n"
                f"# Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# The dataset was made using the following arguments:\n# ")
    comments += ' '.join(f'--{key} {value}' for key, value in vars(args).items())
    comments += '\n'
    comments += f'# dataset_images_hash: {shuffled_images_hash} \n\n'

    data = {
        'path': args.name,  # dataset root dir
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': yaml_label_map
    }

    with open(Path(save_location, filename), 'w') as file:
        file.write(comments)

        yaml.dump(data, file)

    print(f"Created a YAML file in {save_location}.")

# endreion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=Path,
                        help='File path of the data you want to process. This should be the path/to/RDD2022.')
    parser.add_argument('--test-run', action='store_true',
                        help='Dont move images or create labels files (still creates directories)')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='The fraction of the dataset to pre-process, value must between 0 - 1.0')
    parser.add_argument('--move-files', action='store_true', help='Moves images instead of copying images')
    parser.add_argument('--json', action='store_true', help='Use this if the data is in the json format')
    parser.add_argument('-y', '--yes', default=False, action='store_true',
                        help='Automatically answer YES to all questions with this option enabled.')
    parser.add_argument('-n', '--name', type=str, default='RDD2022_PP', help='Give a descriptive (file)name for the preprocessed dataset.')
    parser.add_argument('--exclude', type=str, help='Exclude images from the dataset containing the specified filter terms. '
                                                    'Example usage: --exclude "China_Drone India" filters out all images whose filenames contain China_Drone or India')
    parser.add_argument('-l', '--image-list', type=str, help='Create a preprocessed image set from an image list. Give the file path of a folder with the files train.txt, val.txt and test.txt.')
    args = parser.parse_args()

    print()
    try:
        main(args)
    except KeyboardInterrupt:
        print('\n\n... halting preprocessing')
    print()
