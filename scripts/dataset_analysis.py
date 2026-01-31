import argparse
import os
from pathlib import Path
import json


def main(args):
    
    # UNIX_GROUP = "project_2012962"
    # SCRATCH = f"/scratch/{UNIX_GROUP}"
    # PROJAPPL = f"/projappl/{UNIX_GROUP}"

    dataset_path = f'../datasets/RDD2022_allas'
    
    if not os.path.exists(dataset_path):
        print('No such dataset: "{}"'.format(dataset_path))
        exit(1)
    # dataset_pp_path = f'{SCRATCH}/datasets/RDD2022_allas_pp_{int(args.fraction*100)}'
    
    print('Pre-processing dataset: "{}"'.format(dataset_path))
    print('Using fraction: {:.0f}%'.format(args.fraction*100))
    # print('Saving preprocessed dataset to: "{}"'.format(dataset_pp_path))
    
    train_img_dir = f'{dataset_path}/train/img'
    train_images = [ str(img) for img in Path(train_img_dir).glob('*.jpg') ]
    train_images = [ img for img in train_images if 'China_Drone' not in str(img) ]
    
    print('Found {:_} images in /train/img'.format(len(train_images)))

    anns = []

    for idx, img in enumerate(train_images):
        if idx%1 == 0: print('{:<6}: "{}"'.format(idx, str(img)))
        ann = get_img_ann(img)
        if len(ann.get('objects')) > 0:
            ann = simplify_ann(img, ann)
            anns.append(ann)
            # print('objects:')
            # print(json.dumps(ann,
                    # sort_keys=True, indent=4))
            # break
        # break
    
    
    print('done.')




# region helpers

def simplify_ann(img, ann):
    img_name = img.split(os.sep)[-1]
    obj = {
        'img_name': img_name,
        'size': ann.get('size'),
        'objects': [ simplify_object(obj) for obj in ann.get('objects') ]
    }
    return obj


def simplify_object(obj):
    def get_bbox(points):
        return points
        
    obj_simp = {
        'classTitle': obj.get('classTitle'),
        'bbox': get_bbox(obj['points']['exterior']),
    }
    return obj_simp

def get_img_ann(img):
        json_fp = get_img_json_filepath(img)
        with open(json_fp, 'r') as f:
            return json.load(f)

def get_img_json_filepath(img):
    return str(img).replace(f'{os.sep}img{os.sep}', f'{os.sep}ann{os.sep}') + '.json'




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--something_bool', action='store_true', help='')
    parser.add_argument('--fraction', help='Fraction of dataset to preprocess', type=float, default=1.0)
    
    args = parser.parse_args()
    print()
    main(args)
    print()