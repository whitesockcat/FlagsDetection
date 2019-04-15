# python2 !
# convert
import json
import cv2
import numpy as np
import os
import random
from glob import glob
DATA_DIR     =  '../../video-seg/two/solution/Dataset/Apollo/data/'
PROJECT_DIR  =  '../Apollo/'

DATA_DIR_I  = DATA_DIR    + 'train/'
DATA_DIR_O  = PROJECT_DIR + 'coco_train2014/' # pic save path

# JASON_FILE  = PROJECT_DIR + 'annotations/category.json' # d8
JASON_ANNOTATION = PROJECT_DIR + 'annotations/instances_train2014.json'  # json save path

WIDTH_I  = 3384
HEIGHT_I = 2710
WIDTH_O  = 3384
HEIGHT_O = 2710
y_start  = 0
exits = glob('../Apollo/coco_train2014/'+'*.jpg')
exits = [filename.split('\\')[-1] for filename in exits]
sample_step = 3
##################
### road03_ins ###
##################
# label_name   = ['background', 'Car', 'Bus', 'Truck','SVehicle','Pedestrian','Motorbike','Bicycle','Train','Signal','Signs']
label_list = np.array([0, 33, 39, 38, -1, 36, 34, 35, -1, 81, 83])

# label_name   = ['background', 'person','bicycle','car','motorcycle','bus','truck','tricycle','rider']
# label_color  = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(128,128,255),(255,128,128)]
# with open(PROJECT_DIR + 'annotations/category.json') as json_data:
#     d8 = json.load(json_data)
# print(d8['categories'][0:8])

# label_list = np.array([0, 36, 35, 33, 34, 39, 38, 40, 37])
credit_list= np.array([0,  1,  2,  0,  2,  1,  1,  2,  1])

dir_ids    = []
image_ids  = []
road_ids   = []
record_ids = []

jason_file_cnt = 0
for root, dirs, files in sorted(os.walk(DATA_DIR)):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.json':
            file_id = f.split('.')            
            temp    = file_id[0].split('_')
            image_id= temp[0] + '_' + temp[1]
            camera_id = temp[3]
            #print (image_id[0], image_id[1], image_id[2], image_id[3])
            # print('root', root)
            dir_id = root.split('data/')[1][:-8]
            #print(dir_id, image_id, camera_id, jason_file_cnt, len(files))
            if (camera_id=='6'): #check if the other one exit or not
                root2 = root[:-2] + ' 5'
                f2    = f[:-6]    + '5.json'
                fullpath2 = os.path.join(root2, f2)
                # print(fullpath, fullpath2)
                if os.path.isfile(fullpath2):
                    if 'road03_ins' in fullpath:################ add ##########
                        jason_file_cnt = jason_file_cnt +1
                        temp2 = fullpath.split('\Camera ')[0]
                        # print('temp2=', temp2)
                        record_ids.append(temp2[-9:])
                        temp2 = fullpath.split('\Label')[0]
                        road_ids.append(temp2[-10:])
                        dir_ids.append(dir_id)
                        image_ids.append(image_id)
    # if (jason_file_cnt>3):
    #    break
# print(len(dir_ids), len(image_ids))

"""
#example read one jason file, draw contour
road_id  = 'road03_ins'
record_id= 'Record030'
image_id = '171206_033651918'
camera_id= '5'
file_path = DATA_DIR + road_id + '/ColorImage/' + record_id + '/Camera '+camera_id + '/' + image_id + '_Camera_'+camera_id +'.jpg'
jason_path= DATA_DIR + road_id + '/Label/'      + record_id + '/Camera '+camera_id + '/' + image_id + '_Camera_'+camera_id +'.json' 
print(file_path)
img = cv2.imread(file_path);
cv2.imwrite('test.png', img)

with open(jason_path) as json_data: 
    d = json.load(json_data)
num_obj = len(d['objects'])

for i in range(0, num_obj):
    obj = d['objects'][i]    
    #print(obj['polygons'])
    #print(len(obj['polygons'][0]))
    num_seg = len(obj['polygons'][0])
    seg = obj['polygons'][0]
    #print(seg[0])
    for j in range (0, num_seg-1):
        cv2.line(img,(seg[j][0], seg[j][1]), (seg[j+1][0], seg[j+1][1]), (255,0,0),2)       
    #print(obj['label'])    
cv2.imwrite('test2.png', img)    
"""
#find the histogram of the labels
camera_ids = ['5','6']
hist = np.zeros(256,np.int)
jason_file_cnt = 0

ann_dict = {}
images = []
annotations = [] 
img_ids = 0
ann_id = 0

for dir_id, road_id, record_id, image_id in zip(dir_ids, road_ids, record_ids, image_ids):
    # print('road_id:{}, record_id:{}'.format(road_id, record_id))
    for camera_id in camera_ids:
        jason_path = DATA_DIR + dir_id + '/Camera '+camera_id + '/' + image_id + '_Camera_'+camera_id +'.json'
        file_path  = DATA_DIR_I + road_id + '/ColorImage/' + record_id + '/Camera '+camera_id + '/' + image_id + '_Camera_'+camera_id +'.jpg'        
        # print(dir_id)
        jason_file_cnt = jason_file_cnt +1
        with open(jason_path) as json_data: 
            d = json.load(json_data)
        num_obj = len(d['objects'])  
        credit = 0
        annotations2 = []  
        if (num_obj>0) and ((jason_file_cnt + ((int(camera_id)-4)* sample_step) ) % sample_step == 0):
            #print(jason_file_cnt, camera_id, jason_file_cnt+((int(camera_id)-4)*3))
            image  = {}
            image['id']            = image_id
            img_ids               += 1
            image['width']         = WIDTH_O
            image['height']        = HEIGHT_O
            image['file_name']     = image_id + '_Camera_' + camera_id +'.jpg' 

            # line_top = HEIGHT_I
            # line_bot = 0   
            for i in range(0, num_obj):
                obj = d['objects'][i]  
                label = obj['label']
                #print(label)
                # polys = np.asarray(obj['polygons'][0])

                if (label in label_list): #ignore group
                    
                    ann                 = {}
                    ann['id']           = ann_id
                    ann_id             += 1
                    ann['image_id']     = image['id']
                    ann['iscrowd']      = 0
                    # print('label:', label)
                    idx = np.where(label_list==label)
                    # print(idx)
                    ann['category_id']  = int(idx[0][0])#idx[0][0] == position of list
                    
                    #trim the polygon
                    num_seg = len(obj['polygons'][0])
                    seg     = obj['polygons'][0]

                    # for j in range (0, num_seg):               
                    #     if (line_top>seg[j][1]):
                    #         line_top = seg[j][1]
                    #     if (line_bot<seg[j][1]):
                    #         line_bot = seg[j][1]
                                   
                        #cv2.line(img,(seg[j][0], seg[j][1]), (seg[j+1][0], seg[j+1][1]), (255,0,0),2)                               
                    seg = np.asarray (seg)
                    ann['area']         = int(cv2.contourArea (seg))                 
                    ann['bbox']         = np.array(cv2.boundingRect(seg)).tolist()  
                    poly_list           = []
                    poly_list.append( (seg.ravel()).tolist())
                    ann['segmentation'] = poly_list
                                    
                    if (ann['area'] > 9):
                        annotations2.append(ann)
                        hist[label] = hist[label] +1                      
                        credit = credit + credit_list[idx[0][0]]

                    #print(obj['polygons'])
                    #print(len(obj['polygons'][0]))
            
            #print("=========================================================================")
            if  (credit > 1) : #5
                annotations = annotations + annotations2       
                #print('--', line_bot, line_top, line_bot-line_top)
                offset = 0  #adaptive croping, only check bottom line
                #print(line_bot, HEIGHT_O + y_start)
                # if (line_bot> (HEIGHT_O + y_start)): 
                #     if (line_top>(y_start+64)):
                #         offset = (line_top - y_start - 32) 
                # print(file_path)
                img    = cv2.imread(file_path) 
                img2   = img # [(y_start+offset):(y_start+offset+HEIGHT_O),:]         
                # if (offset>=0):  #correct annotations
                                
                #     for instance in annotations2 : 
                #         instance['bbox'][1] = instance['bbox'][1] - y_start - offset   
                #         num_seg = int(len(instance['segmentation'][0])/2)
                #         for kk in range (0, num_seg):
                #             instance['segmentation'][0][2*kk+1] = instance['segmentation'][0][2*kk+1]- y_start - offset   
                #             if (instance['segmentation'][0][2*kk+1]<0):
                #                 instance['segmentation'][0][2*kk+1] = 0
                #             elif (instance['segmentation'][0][2*kk+1]>=HEIGHT_O):
                #                 instance['segmentation'][0][2*kk+1]  = HEIGHT_O -1       
                                
                        # for kk in range (0, num_seg-1):                        
                        #     y1 = instance['segmentation'][0][2*kk+1]   
                        #     x1 = instance['segmentation'][0][2*kk]
                        #     y2 = instance['segmentation'][0][2*kk+3]   
                        #     x2 = instance['segmentation'][0][2*kk+2]
                            #print(x1,y1,x2,y2)                                         
                            #cv2.line(img2,(x1,y1), (x2, y2),label_color[instance['category_id']-1],2)       
                            #print(obj['label'])
                            #cv2.rectangle(img2,(instance['bbox'][0],instance['bbox'][1]), \
                            #           (instance['bbox'][0]+instance['bbox'][2],instance['bbox'][1]+instance['bbox'][3]), \
                            #           label_color[instance['category_id']-1],2)            
                    #cv2.imwrite('test2'+ str(img_ids) + '.png', img2)    
                # cv2.imwrite(DATA_DIR_O + image_id + '_Camera_' + camera_id + '.jpg', img2,[(cv2.IMWRITE_JPEG_QUALITY),100])           
                
                if image['file_name'] not in exits:
                    print(image['file_name'])
                images.append(image) 

    #else:
    #    print ("pass")
    #if (jason_file_cnt > 200):#31815
    # if jason_file_cnt % 20 == 0:
    #     break

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

# with open(PROJECT_DIR + 'annotations/instances_train2014_short.json', 'w') as outfile:  
#     outfile.write(json.dumps(images))
# with open(JASON_ANNOTATION, 'w') as outfile:  
#     outfile.write(json.dumps(ann_dict))
