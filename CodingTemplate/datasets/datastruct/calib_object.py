import numpy as np


class Calib_Object(object):
    '''
    calib数据格式： P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 ....
                   P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02  ....
                   P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 ....
                   P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 ...
                   R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 ...
                   Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 ...
                   Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 ...

    对应含义： 对于标定数据来说，首先要明确那些设备需要标定数据： KITTI数据集的数据收集平台装配有2个灰度摄像机，2个彩色摄像机，一个激光雷达，
                四个光学镜头，以及一个GPS导航系统
                =》 p0,p1,p2,p3分别表示 左灰度相机，右灰度相机，左彩色相机，右彩色相机的相机内参（3x4）
                R0_rect 为0号相机的修正矩阵
                Tr_velo_to_cam 从点云坐标到相机坐标的矩阵
                Tr_imu_to_velo 从imu到点云坐标的矩阵s
    '''

    def __init__(self, datas):
        self.p2 = np.array(datas[2].strip().split(
            ' ')[1:], dtype=np.float32).reshape(3, 4)
        self.r0 = np.array(datas[4].strip().split(
            ' ')[1:], dtype=np.float32).reshape(3, 3)
        self.v2c = np.array(datas[5].strip().split(
            ' ')[1:], dtype=np.float32).reshape(3, 4)

        # camera intrinsics and extrinsics
        self.cu = self.p2[0, 2]   # 像素坐标u方向的偏移
        self.cv = self.p2[1, 2]   # 像素坐标v方向的偏移
        self.fu = self.p2[0, 0]   # 相机焦距与图像坐标x值的比值
        self.fv = self.p2[1, 1]   # 相机坐标与图像坐标y值的比值
        self.tx = self.p2[0, 3] / (-self.fu)   #
        self.ty = self.p2[1, 3] / (-self.fv)

    def to_str(self):
        print_str = 'p2: %s \n ro: %s \n v2c: %s' % (
            self.p2, self.r0, self.v2c)
        return print_str

    def trans_to_homo(self, coordinate):
        '''
        function: convert corrdinate to homogeneous coordinates
        parameter: coordindate
        return: homogeneous coordinates
        '''
        homo_coor = np.hstack((coordinate, np.ones(
            (coordinate.shape[0], 1), dtype=np.float32)))
        return homo_coor

    def lidar_to_camer(self, pts):
        '''
        function: convert pts coordindate to camer coordinate
        parameter: pts coordinate
        return camer coordinate
        '''
        pts_homo = self.trans_to_homo(pts)
        camer_coor = np.dot(pts_homo,np.dot(self.v2c.T,self.r0.T))
        return camer_coor
    
    def camer_to_image(self, pts):
        '''
        function: convert camer coordinate to image coordinate
        parameter: camer coordinate
        return image coordinate and correspond depth
        '''
        pts_homo = self.trans_to_homo(pts)
        zc_dot_image = np.dot(pts_homo,self.p2.T)
        image_coor = (zc_dot_image[:,0:2].T / pts_homo[:,2]).T
        image_depth = zc_dot_image[0, 2] - self.p2.T[3,2]
        return image_coor, image_depth


