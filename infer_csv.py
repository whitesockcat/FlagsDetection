"""
inference on a image testset and generate CSV file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
from glob import glob
import logging
import os
import sys
import time
import numpy as np
from caffe2.python import workspace
import pycocotools.mask as mask_util
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops()


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes
    

PROJECT_DIR     = '/userhome/flags/'
OUTPUT          = PROJECT_DIR + sys.argv[1] + '/'
OUTPUT_CSV_DIR  = OUTPUT + sys.argv[2] #result_0607_3384_3.csv'
MODEL           = OUTPUT + 'train/coco_2014_train/generalized_rcnn/' + sys.argv[3]#7.pkl'
THRESHOLD       = sys.argv[4]
# submit_list     = PROJECT_DIR + 'submit_example.csv'
# with open(submit_list, 'r') as f:
#     filenames = f.readlines()
#     # lens = len(filenames)
# filenames = [name.strip('\n') for name in filenames]
filenames = glob(PROJECT_DIR + 'test/' +'*.jpg')
# filenames = glob(PROJECT_DIR + 'test/' +'f*.jpg')
# print(filenames)
# print('++++')
filenames2    = []  
labels        = []  
pix_cnt       = [] 
conf          = [] 
encode_pixel  = []
submits = []
label2order = [0, 4, 9, 17, 20, 24, 2, 30, 22, 37, 7, 11, 5, 21, 27, 38, 18, 31, 26, 12, 29,
            32, 8, 33, 14, 40, 19, 16, 20, 13, 1, 35, 28, 36, 3, 10, 25, 34, 23, 39, 6]
def main():
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(PROJECT_DIR + '101.yaml')
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(MODEL)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    frame_cnt = 0

    for filename in filenames:
        # filename = filename.split(',')[0]
        image_path = filename
        frame_cnt = frame_cnt + 1
        im  = cv2.imread(image_path)
        im2 = im
        print(image_path, frame_cnt, (np.array(im)).shape, (np.array(im2)).shape)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im2, None, timers=timers)
        print('!!!!!allbox:-->', len(cls_boxes))
        keypoints=None
        if isinstance(cls_boxes, list):
            cls_boxes, cls_segms, keypoints, classes = convert_from_cls_format(cls_boxes, cls_segms, keypoints)
        # if cls_segms is not None and len(cls_segms) > 0:
        #         masks = mask_util.decode(cls_segms)        
       
        segms = cls_segms        
        boxes = cls_boxes
        print(boxes)
        # submits.append(filename)
        scores = boxes[:, -1]
        print('scores', scores)
        submit = filename# TODO
        if boxes is not None:
            for i in range(len(boxes)):
                bbox  = boxes[i, :4]
                score = boxes[i, -1]
                label = classes[i]
                order = label2order[label]
                if score < THRESHOLD:
                    continue
                x1, y1, x2, y2 = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
                w, h = x2-x1, y2-y1
                
                submit = submit \
                    + '\t(' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h) + ')\t' \
                    + str(order)
        
        submits.append(submit)    

        # if (frame_cnt>3):
        #    break

        # logger.info('Inference time: {:.3f}s'.format(time.time() - t))

    with open(OUTPUT_CSV_DIR, 'w') as f:
        f.write('\n'.join(submits))
        # f.write('\n'.join(str(submit) for submit in submits))   

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    main()