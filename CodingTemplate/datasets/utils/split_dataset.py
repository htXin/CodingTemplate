import os
import numpy as np 
import random

class DataSplit():
    def __init__(self, seed, root_dir, persent):
        filepath = os.path.join(root_dir, 'KITTI_DATASET_ROOT', 'training', 'image_2')
        assert os.path.exists(filepath)
        self.list_data = os.listdir(filepath)
        self.seed = seed
        self.persent = persent

    def split_dataset(self):
        save_path = os.path.join(os.getcwd(), 'data_split_file',"splitedDataSet")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        random.seed(self.seed)
        random.shuffle(self.list_data)

        train_len = int(len(self.list_data)*self.persent[0])
        test_len = int(len(self.list_data)*self.persent[1])
        valid_len = len(self.list_data) - test_len - train_len
        train_data = self.list_data[:train_len]
        test_data = self.list_data[train_len:train_len+test_len]
        valid_data = self.list_data[train_len+test_len:]

        with open(os.path.join(save_path,'train.txt'), 'w+') as f:
            for i in train_data[:-1]:
                f.write(i.split('.')[0]+'\n')
            f.write(train_data[-1].split('.')[0])
            f.close()
        with open(os.path.join(save_path,'test.txt'), 'w+') as f:
            for i in test_data[:-1]:
                f.write(i.split('.')[0]+'\n')
            f.write(test_data[-1].split('.')[0])
            f.close()
        with open(os.path.join(save_path,'valid.txt'), 'w+') as f:
            for i in valid_data[:-1]:
                f.write(i.split('.')[0]+'\n')
            f.write(valid_data[-1].split('.')[0])
            f.close()

if __name__ == "__main__":
    datasplit = DataSplit(1,'d:',[0.5, 0.4])
    datasplit.split_dataset()