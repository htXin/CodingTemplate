from datasets.kittidataset import KittiDataSet
from datasets.utils import kitti_utils
import numpy as np


class GetKittiDataSet(KittiDataSet):
    def __init__(self, root_dir, split="train", npoints = 16384, classes = 'Car'):
        super().__init__(root_dir, split)
        self.sample_idx_list = [int(data) for data in self.data_idx_list]
        self.npoints = npoints
        self.mode = split
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrain', 'Cyclist')
        elif classes == 'Pedestrain':
            self.classes = ('Backgroud', 'Pedestrain')
        elif classes == 'Cyclist':
            self.classes = ('Backgroud', 'Cyclist')

    @staticmethod
    def get_valid_flag(pts_img, pts_image_depth, image_shape):
        flag_u = np.logical_and(pts_img[:,0]>0, pts_img[:,0]<image_shape[1])
        flag_v = np.logical_and(pts_img[:,1]>0, pts_img[:,1]<image_shape[0])
        flag_merge = np.logical_and(flag_u,flag_v)
        flag_depth = np.logical_and(flag_merge, pts_image_depth>=0)
        return flag_depth
    def filter_objects(self, object_list):
        '''
        dicard the object not in self.classes
        '''
        classes_list = self.classes
        if self.mode == 'trian':
            classes_list = list(self.classes)
            if 'Car' in self.classes:
                classes_list.append('Van')
            elif 'Pedestrain' in self.classes:
                classes_list.append('Person_siting')
        filtered_object = []  
        for obj in object_list:
            if obj.cls_type not in classes_list:
                continue
            filtered_object.append(obj)
        return filtered_object

    def __getitem__(self, index):
        sampleid = self.sample_idx_list[index]
        img = self.get_image_rgb_norm(sampleid)
        calib = self.get_calib(sampleid)
        lidar = self.get_lidar(sampleid)
        label = self.get_label(sampleid)
        image_shape = self.get_image_shape(sampleid)


        pts_camer = calib.lidar_to_camer(lidar[:,0:3])
        pts_indensity = lidar[:,3]
        pts_image,pts_image_depth = calib.camer_to_image(pts_camer)
        pts_flag = self.get_valid_flag(pts_image,pts_image_depth,image_shape)
        
        #get points coordinate(in image range)、indensity and corrsponding image coordinate 
        pts_camer = pts_camer[pts_flag]
        pts_indensity = pts_indensity[pts_flag]
        pts_image = pts_image[pts_flag]
        
        #select points
        if self.npoints <= len(pts_camer):
            pts_depth = pts_camer[:,2]
            depth_flag = pts_depth < 40.0
            far_pts_idx = np.where(depth_flag == 0)[0]
            near_pts_idx = np.where(depth_flag == 1)[0]
            choice = np.random.choice(near_pts_idx, self.npoints-len(far_pts_idx), replace=False)
            choice = np.concatenate(((choice, far_pts_idx)), axis=0) if len(far_pts_idx) > 0 else choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0,len(pts_camer),dtype=np.int32)
            extra_choice = np.random.choice(choice,self.npoints-len(pts_camer),replace=False)
            choice = np.concatenate((choice,extra_choice),axis=0)
            np.random.shuffle(choice)

        pts_camer = pts_camer[choice,:]
        pts_indensity = pts_indensity[choice]
        pts_image = pts_image[choice,:]
        sample_info = {'img': img,
                    #    'label': label,
                       'data_id': sampleid,
                       'image_shape': image_shape,
                       'pts_camer' : pts_camer,
                       'pts_image': pts_image,
                       'pts_indensity' : pts_indensity}

        get_object_list = self.filter_objects(label)
        object_boxes3d = kitti_utils.object_boxes3d(get_object_list)
        sample_info['object_boxes3d'] = object_boxes3d

        return sample_info

    def __len__(self):
        return len(self.sample_idx_list)

    def collate_fun(self, batch):
        batch_size = batch.__len__()
        sample_dict = {}

        for key in batch[0].keys():
            if key == 'object_boxes3d':
                max_obnu = 0
                for k in range(batch_size):
                    max_obnu = max(max_obnu, batch[k][key].__len__())
                batch_object_boxes3d = np.zeros((batch_size, max_obnu, 7), dtype=np.float32)
                for k in range(batch_size):
                    batch_object_boxes3d[k, :batch[k][key].__len__(), :] = batch[k][key]
                sample_dict[key] = batch_object_boxes3d
                continue
            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    sample_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    sample_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)
            else:
                sample_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch_size[0][key], int):
                    sample_dict[key] = np.array(sample_dict[key], np.int32)
                elif isinstance(batch_size[0][key], float):
                    sample_dict[key] = np.array(sample_dict[key], np.float32)
        return sample_dict
if __name__ == '__main__':
    dataset = GetKittiDataSet('d:')
    data = dataset.__getitem__(0)
    print(len(data['pts_camer']))


