import json
import cv2
import numpy as np

DATA_DIR     =  '../dtc_train_images/'
JASON_OUT = '../json_out/instances_train2014.json'  # json save path

label_name   = np.array(['Car', 'Bus', 'Truck','SVehicle','Pedestrian','Motorbike','Bicycle','Train','Signal','Signs'])

WIDTH_I  = 1936
HEIGHT_I = 1216
HEIGHT_O = 950

img_ids = 0
ann_id = 0
img_cnt = 0
ann_dict = {}
images = []
annotations = [] 

for i in range(21257+1):
    jason_path  = '../dtc_train_annotations/train_' + '{0:05}'.format(i) + '.json'
    img_path    = '../dtc_train_images/train_'      + '{0:05}'.format(i) + '.jpg'

    with open(jason_path) as json_data: 
        json_in = json.load(json_data)
    num_obj = len(json_in['labels'])

    if num_obj > 0:
        image  = {}
        image['id']            = 'train_{0:05}'.format(i)
        img_ids               += 1
        image['width']         = WIDTH_I
        image['height']        = HEIGHT_I
        image['file_name']     = 'train_{0:05}'.format(i) + '.jpg'

        annotations2 = []
        
        for i in range(num_obj):
            obj = json_in['labels'][i]
            label = obj['category']
                
            ann                 = {}
            ann['id']           = ann_id
            ann_id             += 1
            ann['image_id']     = image['id']
            ann['iscrowd']      = 0
            idx = np.where(label_name==label)
            print('label:', label)
            print(idx)
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
