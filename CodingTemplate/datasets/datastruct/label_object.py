import numpy as np


def cls_type_to_id(cls_type):
    typewithid = {'Car': 1, 'Pedestrain': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in typewithid.keys():
        return -1
    return typewithid[cls_type]


class Label_Object(object):
    def __init__(self, data):
        """
        label 数据格式： Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
        对应含义         1.类别 
                        2.是否被截断 0-1之间浮动
                        3.遮挡 0-3 分别表示 完全可见，小部分遮挡， 大部分遮挡， 完全遮挡
                        4.alpha,物体的观察角度 -pi~pi 
                        5~8.物体的二维边界框（像素） xmin，ymin，xmax，ymax 
                        9~11.物体的三维尺寸（米）高宽长
                        12~14.物体的三维位置（米）xyz
                        15.三维物体的空间方向 rotation_y
        """
        self.src = data
        label = data.strip().split(' ')
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])
        self.occlusion = float(label[2])
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(
            label[12]), float(label[13])), dtype=np.float32)
        self.dist_to_cam = np.linalg.norm(self.pos)  # 计算物体到相机的距离
        self.rotation_y = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -0.1
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4
    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.pos, self.rotation_y)
        return print_str 