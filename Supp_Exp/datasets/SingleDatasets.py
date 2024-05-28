import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HADTestDataset(Dataset):
    def __init__(self,
                 dataset_path='./',
                 start_channel=0,
                 channel = 100,
                 scene = ''
                 ):
        self.dataset_path = dataset_path
        self.start_channel = start_channel
        self.channel =channel
        self.scene = scene

        self.test_img, self.gt_img= self.load_dataset_folder()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, idx):
        x, gt= self.test_img[idx], self.gt_img[idx]
        x = np.load(x)
        x = x[:, :, self.start_channel:(self.channel + self.start_channel)]
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        x = x * 0.1
        x = self.transform(x)
        x = x.type(torch.FloatTensor)

        gt = Image.open(gt)
        gt =np.array(gt)
        gt = Image.fromarray(gt)
        gt = self.transform(gt)
        return x,gt

    def __len__(self):
        return len(self.test_img)

    def load_dataset_folder(self):
        test_img_dir = os.path.join(self.dataset_path, 'test/', self.scene)
        gt_dir = os.path.join(self.dataset_path, 'ground_truth/',self.scene)
        test_img = sorted(
            [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.npy')])
        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in test_img]
        gt_img = [os.path.join(gt_dir, img_name + '.png') for img_name in img_name_list]
        assert len(test_img) == len(gt_img), 'number of test img and gt should be same'
        return test_img, gt_img


