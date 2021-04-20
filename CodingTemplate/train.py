import numpy as np
import argparse
from datasets import get_kittidataset
from torch.utils.data import DataLoader
from net import MNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

parse = argparse.ArgumentParser("xxnet trian parse")
parse.add_argument("--dataset_path", type=str, default='d:')
parse.add_argument("--model", type=str, default='train',
                   help='selcet model is trian or test')
parse.add_argument('--batch_size', type=int, default=8,
                   help='dataset size of each batch')
parse.add_argument('--train_lr', type=int, default=1e-3, help='learning rate of training')
args = parse.parse_args()


def getDataloader(dataset_path):
    dataset = get_kittidataset.GetKittiDataSet(dataset_path, split=args.model)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True,
                            drop_last=True, collate_fn=dataset.collate_fun)
    return dataloader


def create_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=args.train_lr)
    return optimizer


if __name__ == '__main__':
    dataloader = getDataloader(args.dataset_path)
    # for i, data in enumerate(dataloader):
    #     print(data)
    #     break
    model = MNet.MNet()
    optimizer = create_optimizer(model)
