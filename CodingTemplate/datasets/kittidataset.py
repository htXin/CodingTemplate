import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import datasets.datastruct.label_object as label_object
import datasets.datastruct.calib_object as calib_object
import argparse


class KittiDataSet(Dataset):
    def __init__(self, root_dir, split):
        IS_TRAIN = split == 'train'
 

        self.data_dir = os.path.join(
            root_dir, 'KITTI_DATASET_ROOT', 'training' if IS_TRAIN else 'testing')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.lidar_dir = os.path.join(self.data_dir, 'velodyne')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        split_dir = os.path.join(
            os.getcwd(), "data_split_file", 'splitedDataSet', split+'.txt')
        # print(split_dir)
        assert os.path.exists(split_dir)
        self.data_idx_list = [line.strip()
                              for line in open(split_dir).readlines()]
        self.num_data = self.data_idx_list.__len__()

    def get_data_idx_list(self):
        return self.data_idx_list

    def get_image(self, index):
        import cv2
        imagefile = os.path.join(self.image_dir, '%06d.png' % index)
        assert os.path.exists(imagefile)
        return cv2.imread(imagefile)

    def get_image_shape(self, index):
        imagefile = os.path.join(self.image_dir, '%06d.png' % index)
        assert os.path.exists(imagefile)
        image = Image.open(imagefile)
        w, h = image.size
        return h, w, 3

    def get_image_rgb_norm(self, index):
        """
        return image with normalziation in rgb model
        param: index
        return image(H,W,3)
        """
        imagefile = os.path.join(self.image_dir, '%06d.png' % index)
        # print(imagefile)
        assert os.path.exists(imagefile)
        img = Image.open(imagefile).convert('RGB')
        img = np.array(img).astype(np.float)
        img = img / 255.0
        img -= self.mean
        img /= self.std
        imback = np.zeros([384, 1280, 3], dtype=np.float)
        imback[:img.shape[0], :img.shape[1], :] = img
        return imback

    def get_lidar(self, index):
        """
        bin 文件存储点云数据方式：
            x1,y1,z1,r1,x2,y2,z2,r2,.......xi,yi,zi,ri
        其中xi，yi，zi，ri 表示点云数据i的坐标：xyz以及反射率r
        """
        liadrfile = os.path.join(self.lidar_dir, '%06d.bin' % index)
        assert os.path.exists(liadrfile)
        return np.fromfile(liadrfile, dtype=np.float32).reshape(-1, 4)

    def get_label(self, index):
        """
        return label of image and lidar
        param: index
        """
        labelfile = os.path.join(self.label_dir, '%06d.txt' % index)
        assert os.path.exists(labelfile)
        with open(labelfile, 'r') as f:
            lines = f.readlines()
        objects = list([label_object.Label_Object(line) for line in lines])
        return objects

    def get_calib(self, index):
        '''
        return calib of each image 
        param: index
        '''
        calibfile = os.path.join(self.calib_dir, '%06d.txt' % index)
        assert os.path.exists(calibfile)
        with open(calibfile, 'r') as f:
            lines = f.readlines()
        calibinfo = calib_object.Calib_Object(lines)
        return calibinfo

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='get kittidataset root')
    args.add_argument('--data_set_root', type=str,
                      default='d:', help='kittiDataSet root')
    args.add_argument('--model', type=str, default='train',
                      help='select model(such as ''trian'')')
    arg = args.parse_args()

    root_dir = arg.data_set_root
    dataset = KittiDataSet(root_dir, arg.model)
    img = dataset.get_image_rgb_norm(000000)
    # img2 = dataset.get_image(000000)
    # import cv2
    # cv2.imshow('image', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)
    lidar = dataset.get_lidar(000000)
    label = dataset.get_label(000000)
    calib = dataset.get_calib(000000)
    print(lidar.shape)
    print(label[0].to_str())
    print(calib.to_str())
    print(len(dataset.get_data_idx_list()))
