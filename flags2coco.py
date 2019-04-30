import json
import cv2
import numpy as np
import glob, os
import random


PROJECT_DIR  =  'flag2coco/'

DATA_DIR_O   = PROJECT_DIR + 'coco_train2014/' # pic save path
ANN_DIR_O    = PROJECT_DIR + 'annotations/'
JASON_ANNOTATION = PROJECT_DIR + 'annotations/instances_train2014.json'  # json save path
if not os.path.exists(DATA_DIR_O):
    os.makedirs(DATA_DIR_O)
if not os.path.exists(ANN_DIR_O):
    os.makedirs(ANN_DIR_O)

label_name   = np.array(['background', 'meiguo', 'hanguo', 'baxi','jianpuzhai',
                        'xinjiapo','dangqi','libanen','taiguo','baieluosi',
                        'riben', 'xibanya', 'yingguo', 'miandian', 'yilang',
                        'beiyue', 'yuenan', 'yiselie', 'yilake', 'aoyunhui',
                        'yuedan', 'balesitan', 'chaoxian', 'shatealabo', 'oumeng',
                        'shimao', 'laowo', 'yindu', 'feilvbin', 'lianheguo',
                        'zhongguo', 'aodaliya', 'xuliya', 'jianada', 'bayijunqi',
                        'eluosi', 'afuhan', 'ruidian', 'malaixiya', 'dongnanyalianmeng',
                        'faguo'])

train_labels     = 'label.csv'
with open(train_labels, 'r') as f:
    filenames = f.readlines()
    # lens = len(filenames)
filenames = [name.strip('\n') for name in filenames]

name_bboxes = {}
name_label = {}
img_names = []
for name_box in filenames:
    if 'jpg' in name_box:
        element = name_box.split(',')
        name = element[0]
        xmin, ymin, xmax, ymax = int(element[1]), int(element[2]), \
                                int(element[3]), int(element[4])
        label = element[-1]
        # 一张只有一个 不用判断了
        name_label[name] = label
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

num_pic = 206000
for i in range(num_pic):
    name = img_names[i] # ./mergePic/eluosi_merge_5050.jpg
    img_full_path = 'flag2coco/coco_train2014/' + name[11:]
    img    = cv2.imread(img_full_path)
    height, width, channels = img.shape
    image  = {}
    image['id']            = name[11:-4]
    img_ids               += 1
    image['width']         = width
    image['height']        = height
    image['file_name']     = name[11:]
    annotations2 = []
    bboxes = name_bboxes[name]
    label = name_label[name]
    num_obj = len(bboxes)
    if num_obj:
        for i in range(num_obj):
            ann                 = {}
            ann['id']           = ann_id
            ann_id             += 1
            ann['image_id']     = image['id']
            ann['iscrowd']      = 0
            idx = np.where(label_name==label)
            ann['category_id']  = int(idx[0][0])

            bbox = bboxes[i]
            ann['area']         = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ann['bbox']         = [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
            poly_list           = [bbox[0],     (bbox[1] + bbox[3])/2, 
                                    (bbox[0] + bbox[2])/2,    bbox[1], 
                                    bbox[2],    (bbox[1] + bbox[3])/2, 
                                    (bbox[0] + bbox[2])/2,    bbox[3]]
            ann['segmentation'] = [poly_list]
            annotations2.append(ann)

        annotations = annotations + annotations2
        # cv2.imwrite(DATA_DIR_O + name[-5:], img,[(cv2.IMWRITE_JPEG_QUALITY),100])
        images.append(image)

    # img_cnt += 1
    # if img_cnt % 20 == 0:
    #     break


CATEGORIES = [
    {
        'id': 1,
        'name': 'meiguo',
        'supercategory': 'flag',
    },
    {
        'id': 2,
        'name': 'hanguo',
        'supercategory': 'flag',
    },
    {
        'id': 3,
        'name': 'baxi',
        'supercategory': 'flag',
    },
    {
        'id': 4,
        'name': 'jianpuzhai',
        'supercategory': 'flag',
    },
    {
        'id': 5,
        'name': 'xinjiapo',
        'supercategory': 'flag',
    },
    {
        'id': 6,
        'name': 'dangqi',
        'supercategory': 'flag',
    },
    {
        'id': 7,
        'name': 'libanen',
        'supercategory': 'flag',
    },
    {
        'id': 8,
        'name': 'taiguo',
        'supercategory': 'flag',
    },
    {
        'id': 9,
        'name': 'baieluosi',
        'supercategory': 'flag',
    },
    {
        'id': 10,
        'name': 'riben',
        'supercategory': 'flag',
    },
    {
        'id': 11,
        'name': 'xibanya',
        'supercategory': 'flag',
    },
    {
        'id': 12,
        'name': 'yingguo',
        'supercategory': 'flag',
    },
    {
        'id': 13,
        'name': 'miandian',
        'supercategory': 'flag',
    },
    {
        'id': 14,
        'name': 'yilang',
        'supercategory': 'flag',
    },
    {
        'id': 15,
        'name': 'beiyue',
        'supercategory': 'flag',
    },
    {
        'id': 16,
        'name': 'yuenan',
        'supercategory': 'flag',
    },
    {
        'id': 17,
        'name': 'yiselie',
        'supercategory': 'flag',
    },
    {
        'id': 18,
        'name': 'yilake',
        'supercategory': 'flag',
    },
    {
        'id': 19,
        'name': 'aoyunhui',
        'supercategory': 'flag',
    },
    {
        'id': 20,
        'name': 'yuedan',
        'supercategory': 'flag',
    },
    {
        'id': 21,
        'name': 'balesitan',
        'supercategory': 'flag',
    },
    {
        'id': 22,
        'name': 'chaoxian',
        'supercategory': 'flag',
    },
    {
        'id': 23,
        'name': 'shatealabo',
        'supercategory': 'flag',
    },
    {
        'id': 24,
        'name': 'oumeng',
        'supercategory': 'flag',
    },
    {
        'id': 25,
        'name': 'shimao',
        'supercategory': 'flag',
    },
    {
        'id': 26,
        'name': 'laowo',
        'supercategory': 'flag',
    },
    {
        'id': 27,
        'name': 'yindu',
        'supercategory': 'flag',
    },
    {
        'id': 28,
        'name': 'feilvbin',
        'supercategory': 'flag',
    },
    {
        'id': 29,
        'name': 'lianheguo',
        'supercategory': 'flag',
    },
    {
        'id': 30,
        'name': 'zhongguo',
        'supercategory': 'flag',
    },
    {
        'id': 31,
        'name': 'aodaliya',
        'supercategory': 'flag',
    },
    {
        'id': 32,
        'name': 'xuliya',
        'supercategory': 'flag',
    },
    {
        'id': 33,
        'name': 'jianada',
        'supercategory': 'flag',
    },
    {
        'id': 34,
        'name': 'bayijunqi',
        'supercategory': 'flag',
    },
    {
        'id': 35,
        'name': 'eluosi',
        'supercategory': 'flag',
    },
    {
        'id': 36,
        'name': 'afuhan',
        'supercategory': 'flag',
    },
    {
        'id': 37,
        'name': 'ruidian',
        'supercategory': 'flag',
    },
    {
        'id': 38,
        'name': 'malaixiya',
        'supercategory': 'flag',
    },
    {
        'id': 39,
        'name': 'dongnanyalianmeng',
        'supercategory': 'flag',
    },
    {
        'id': 40,
        'name': 'faguo',
        'supercategory': 'flag',
    }
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
