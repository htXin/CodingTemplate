import numpy as np 
import torch 

def object_boxes3d(object_list):
    boxes3d = np.zeros((object_list.__len__(), 7), dtype = np.float32)
    for k, obj in enumerate(object_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.pos, obj.h, obj.w, obj.l, obj.rotation_y
    return boxes3d