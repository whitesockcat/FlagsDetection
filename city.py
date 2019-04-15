# python2 !
# convert
import json
import cv2
import numpy as np
import glob, os
import random

DATA_DIR     =  '../'
PROJECT_DIR  =  '../city/'
JSON_DIR     =  DATA_DIR   + '../video-seg/two/solution/gtFine_trainvaltest/gtFine/'
DATA_DIR_I   = DATA_DIR    + '../video-seg/two/solution/leftImg8bit_trainvaltest/leftImg8bit/'
DATA_DIR_O   = PROJECT_DIR + 'train_2048/' # pic save path

# JASON_FILE  = PROJECT_DIR + 'annotations/category.json' # d8
JASON_ANNOTATION = PROJECT_DIR + 'annotations/train_mini.json'  # json save path

WIDTH_I  = 2048
HEIGHT_I = 1024
WIDTH_O  = WIDTH_I
HEIGHT_O = HEIGHT_I
y_start  = 0 


label_name   = np.array(['background', 'car', 'bus', 'truck','NO-SVehicle','person','motorcycle','bicycle','NO-Train','traffic light','traffic sign'])


dir_ids    = []
image_ids  = []

jason_file_cnt = 0
for root, dirs, files in sorted(os.walk(JSON_DIR)):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.json':
            file_id = f.split('.')
            len_houzhui = len('_gtFine_polygons')
            image_id = file_id[0][:-len_houzhui]
            # temp    = file_id[0].split('_')
            dir_id = root.split('gtFine/')[1]# eg: train/aachen
            #print(dir_id, image_id, camera_id, jason_file_cnt, len(files))
            
            dir_ids.append(dir_id)
            image_ids.append(image_id)
    #if (jason_file_cnt>3):
    #    break
# print(len(dir_ids), len(image_ids))


jason_file_cnt = 0

ann_dict = {}
images = []
annotations = [] 
img_ids = 0
ann_id = 0

for dir_id, image_id in zip(dir_ids, image_ids):
    jason_path = JSON_DIR + dir_id  + '/' + image_id + '_gtFine_polygons.json'
    file_path  = DATA_DIR_I + dir_id + '/' + image_id + '_leftImg8bit.png'        
    # print(file_path)
    jason_file_cnt = jason_file_cnt +1
    with open(jason_path) as json_data: 
        d = json.load(json_data)
    num_obj = len(d['objects'])  
    credit = 0   
    annotations2 = []  
    if (num_obj>0):
        #print(jason_file_cnt, camera_id, jason_file_cnt+((int(camera_id)-4)*3))
        image  = {}
        image['id']            = image_id
        img_ids               += 1
        image['width']         = WIDTH_O
        image['height']        = HEIGHT_O
        image['file_name']     = image_id + '.jpg'

        line_top = HEIGHT_I
        line_bot = 0   
        for i in range(0, num_obj):
            obj = d['objects'][i]  
            label = obj['label']
            #print(label)
            polys = np.asarray (obj['polygon'][0])

            if (label not in label_name or label == 'background'):
                continue
                
            ann                 = {}
            ann['id']           = ann_id
            ann_id             += 1
            ann['image_id']     = image['id']
            ann['iscrowd']      = 0
            idx = np.where(label_name==label)
            ann['category_id']  = int(idx[0][0])
            
            #trim the polygon
            num_seg = len(obj['polygon'])
            seg     = obj['polygon']
            # print('num_seg:', num_seg)
            seg = np.asarray (seg)
            ann['area']         = int(cv2.contourArea (seg))
            ann['bbox']         = np.array(cv2.boundingRect(seg)).tolist()  
            poly_list           = []
            poly_list.append( (seg.ravel()).tolist())
            ann['segmentation'] = poly_list
                            
            annotations2.append(ann)

        #print("=========================================================================")
        annotations = annotations + annotations2
        img    = cv2.imread(file_path)
        # cv2.imwrite(DATA_DIR_O + image_id + '.jpg', img,[(cv2.IMWRITE_JPEG_QUALITY),100])
        images.append(image) 

    #else:
    #    print ("pass")
    #if (jason_file_cnt > 200):#31815
    if jason_file_cnt % 20 == 0:
        break

    if jason_file_cnt % 200 == 0:
        print(len(images),' slected files', len(annotations), 'annotations', ' Processed ', jason_file_cnt, ' jason files')        
    # break         
# print(hist)

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

with open(JASON_ANNOTATION, 'w') as outfile:  
    outfile.write(json.dumps(ann_dict))
