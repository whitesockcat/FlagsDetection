import json
jason_path = 'flag2coco/annotations/instances_train2014.json'
with open(jason_path) as json_data: 
    json_in = json.load(json_data)
images = json_in['images']
categories = json_in['categories']
anns = json_in['annotations']
# print(len(anns))
anns_add = []
for ann in anns:
    if ann['area'] < 0:
        print(ann['area'])
        print(ann['image_id'])
    # x1, y1, w, h = ann['bbox']
    # poly_list   = [x1,     y1, 
    #             x1 + w,   y1, 
    #             x1 + w,    y1 + h, 
    #             x1,   y1 + h]
    # ann['segmentation'] = [poly_list]
    # anns_add.append(ann)

ann_dict = {}
ann_dict['images']      = images
ann_dict['categories']  = categories
ann_dict['annotations'] = anns_add

# with open('instances_train2014.json', 'w') as outfile:  
#     outfile.write(json.dumps(ann_dict))
