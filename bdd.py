import json
import cv2
import numpy as np
import glob, os
import random


JASON_OUT = '../bdd_coco/annotations/instances_train2014.json'  # json save path
IMG_OUT   = '../bdd_coco/coco_train2014/'
label_name   = np.array(['car', 'bus', 'truck','NO-SVehicle','person','motorcycle','bicycle','train','traffic light','traffic sign'])

WIDTH_I  = 1280
HEIGHT_I = 720

img_ids = 0
ann_id = 0
img_cnt = 0
ann_dict = {}
images = []
annotations = [] 

# train.json 100k/train pic_num 69863 not all
# val.json   10k/train  pic_num 10000     all
jason_path = '../bdd100k/labels/bdd100k_labels_images_val.json'
# jason_path = '../bdd100k/labels/bdd100k_labels_images_train.json'
with open(jason_path) as json_data: 
    json_in = json.load(json_data)
pic_num = len(json_in)

for i in range(pic_num):
    pic = json_in[i]
    name = pic['name']
    img_path    = '../bdd100k/images/10k/train/' + name
    # img_path    = '../bdd100k/images/100k/train/' + name
    labels = pic['labels']

    image  = {}
    image['id']            = name[:-4] # remove .jpg
    img_ids               += 1
    image['width']         = WIDTH_I
    image['height']        = HEIGHT_I
    image['file_name']     = name

    annotations2 = []
    num_obj = len(labels)
    if num_obj:
        img = cv2.imread(img_path) 
        cv2.imwrite(IMG_OUT + name, img,[(cv2.IMWRITE_JPEG_QUALITY),100])           

        for j in range(num_obj):
            obj = labels[j]
            label = obj['category']
            
            if label not in label_name:
                continue
            ann                 = {}
            ann['id']           = ann_id
            ann_id             += 1
            ann['image_id']     = image['id']
            ann['iscrowd']      = 0
            idx = np.where(label_name==label)
            # print('label:', label)
            # print(idx)
            ann['category_id']  = int(idx[0][0]) + 1 #idx[0][0] == position of list

            bbox = obj['box2d']
            
            width = bbox['x2'] - bbox['x1']
            hight = bbox['y2'] - bbox['y1']
            ann['area']         = width * hight
            ann['bbox']         = [bbox['x1'], bbox['y1'], width, hight]
            poly_list           = [bbox['x1'],  bbox['y1'], 
                                    bbox['x2'], bbox['y1'], 
                                    bbox['x2'], bbox['y2'], 
                                    bbox['x1'], bbox['y2'],]
            ann['segmentation'] = [poly_list]
            annotations2.append(ann)

        annotations = annotations + annotations2
        images.append(image)

        # img_cnt += 1
        # if img_cnt % 20 == 0:
        #     break

CATEGORIES = [
    {
        'id': 1,
        'name': 'Car',
        'supercategory': 'vehicle',
    },
    {
        'id': 2,
        'name': 'Bus',
        'supercategory': 'vehicle',
    },
    {
        'id': 3,
        'name': 'Truck',
        'supercategory': 'vehicle',
    },
    {
        'id': 4,
        'name': 'SVehicle',
        'supercategory': 'vehicle',
    },
    {
        'id': 5,
        'name': 'Pedestrian',
        'supercategory': 'person',
    },
    {
        'id': 6,
        'name': 'Motorbike',
        'supercategory': 'vehicle',
    },
    {
        'id': 7,
        'name': 'Bicycle',
        'supercategory': 'vehicle',
    },
    {
        'id': 8,
        'name': 'Train',
        'supercategory': 'vehicle',
    },
    {
        'id': 9,
        'name': 'Signal',
        'supercategory': 'sign',
    },
    {
        'id': 10,
        'name': 'Signs',
        'supercategory': 'sign',
    },
]

ann_dict['images']      = images
ann_dict['categories']  = CATEGORIES    # d8['categories']
ann_dict['annotations'] = annotations
print("Num categories: %s" % len(CATEGORIES))
print("Num images: %s" % len(images))
print("Num annotations: %s" % len(annotations))
# print('annotations ', type(annotations[0]['category_id']))

with open(JASON_OUT, 'w') as outfile:  
    outfile.write(json.dumps(ann_dict))
