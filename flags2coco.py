import json
import cv2
import numpy as np
import glob, os
import random


PROJECT_DIR  =  'flags/'

DATA_DIR_O   = PROJECT_DIR + 'coco_train2014/' # pic save path
ANN_DIR_O    = PROJECT_DIR + 'annotations/'
JASON_ANNOTATION = PROJECT_DIR + 'annotations/instances_train2014.json'  # json save path
if not os.path.exists(DATA_DIR_O):
    os.makedirs(DATA_DIR_O)
if not os.path.exists(ANN_DIR_O):
    os.makedirs(ANN_DIR_O)
    
train_labels     = 'China_flag.csv'
with open(train_labels, 'r') as f:
    filenames = f.readlines()
    # lens = len(filenames)
filenames = [name.strip('\n') for name in filenames]

name_bboxes = {}
img_names = []
for name_box in filenames:
    if 'jpg' in name_box:
        element = name_box.split(',')
        name = element[0]
        xmin, ymin, xmax, ymax = int(element[1]), int(element[2]), \
                                int(element[3]), int(element[4])
        if name in name_bboxes:
            name_bboxes[name].append([xmin, ymin, xmax, ymax])
        else:
            name_bboxes[name] = [[xmin, ymin, xmax, ymax]]
        if name not in img_names:
            img_names.append(name)
# print('name_bboxes:', name_bboxes)
# print('names :', img_names)

ann_dict = {}
images = []
annotations = [] 
img_ids = 0
ann_id = 0
img_cnt = 0
import pandas as pd
import numpy as np

num_pic = 5
for i in range(num_pic):
    name = img_names[i]
    img_full_path = name[8:]
    img    = cv2.imread(img_full_path)
    height, width, channels = img.shape
    image  = {}
    image['id']            = name[-5:] # remove '.jpg'
    img_ids               += 1
    image['width']         = width
    image['height']        = height
    image['file_name']     = name
    annotations2 = []
    bboxes = name_bboxes[name]
    num_obj = len(bboxes)
    if num_obj:
        for i in range(num_obj):

            ann                 = {}
            ann['id']           = ann_id
            ann_id             += 1
            ann['image_id']     = image['id']
            ann['iscrowd']      = 0
            ann['category_id']  = 1 # 只支持一种 这不行

            bbox = bboxes[i]
            ann['area']         = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ann['bbox']         = [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
            annotations2.append(ann)

        annotations = annotations + annotations2
        cv2.imwrite(DATA_DIR_O + name[-5:], img,[(cv2.IMWRITE_JPEG_QUALITY),100])
        images.append(image)

    # img_cnt += 1
    # if img_cnt % 20 == 0:
    #     break

CATEGORIES = [
    {
        'id': 1,
        'name': 'china',
        'supercategory': 'flag',
    },
]

ann_dict['images']      = images
ann_dict['categories']  = CATEGORIES
ann_dict['annotations'] = annotations
print("Num categories: %s" % len(CATEGORIES))
print("Num images: %s" % len(images))
print("Num annotations: %s" % len(annotations))
# print('annotations ', type(annotations[0]['category_id']))

with open(JASON_ANNOTATION, 'w') as outfile:  
    outfile.write(json.dumps(ann_dict))
