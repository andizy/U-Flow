import torch.nn.functional as F
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torchvision
from torchvision.datasets import ImageFolder
from torchdata.datapipes.map import SequenceWrapper
from torch.utils.data import random_split


class scattering_dataloader(torch.utils.data.Dataset):


    def __init__(self, directory,typei='gt' ):
        super(scattering_dataloader, self).__init__()

        self.directory= directory
        self.name_list = os.listdir(self.directory)

        self.typei = typei

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        file_name = self.name_list[idx]
        sample = np.load(os.path.join(self.directory,file_name))

        if self.typei == 'gt':

            uu = np.expand_dims(sample , axis = 0)
            uu = (uu-1540.0)/(5750.0-1540.0)
            sampli = torch.tensor(uu, dtype = torch.float32)

        elif self.typei == 'measure':

            sampli = np.zeros((2,np.shape(sample)[0],np.shape(sample)[1]))
            sampli[0]= sample.real
            sampli[1] = sample.imag
            sampli = torch.tensor(sampli/400000.0, dtype = torch.float32)
            sampli = F.interpolate(sampli[None, ...], 128 , mode = 'bilinear', align_corners=False)[0]
            # sampli = sampli.repeat(1,8,8)

        elif self.typei[0:2] == 'bp':
            uu = np.expand_dims(sample , axis = 0)
            uu = (uu-(-2e-6))/(2e-6-(-2e-6))
            sampli = torch.tensor(uu, dtype = torch.float32)            

        return sampli
        


class general_dataloader(torch.utils.data.Dataset):
    def __init__(self, dataset = 'mnist', size=(32,32), c = 1):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        self.c = c
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.img_dataset = torchvision.datasets.MNIST('data/MNIST', train=True,
                                                    download=True)
        
        elif self.dataset == 'celeba-hq':
            celeba_path = '/raid/Amir/Projects/datasets/celeba_hq/celeba_hq_256/'
            self.img_dataset = ImageFolder(celeba_path, self.transform)
            
            

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        if self.dataset == 'celeba-hq':
            img = transforms.ToPILImage()(img)

        img = self.transform(img)

        return img
    
class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, 
        size=(32,32), 
        c = 1,
        missing_cone = 'horizontal',
        cond = False,
        ):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            
        ])

        self.c = c
        self.transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])
        x_limited_ct_path = '/raid/Amir/Projects/datasets/CT_dataset/images/gt_train'
        self.img_dataset = ImageFolder(x_limited_ct_path, self.transform)
        if cond:
            if missing_cone == "vertical":
                y_folder = "/raid/Amir/Projects/datasets/CT_dataset/images/fbp_train_vertical_snr_40"
            else:
                y_folder = "/raid/Amir/Projects/datasets/CT_dataset/images/fbp_train_horizontal_snr_40"
            self.img_dataset = ImageFolder(y_folder, self.transform)

            

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        img = transforms.ToPILImage()(img)
        img = self.transform(img)
        return img  

def load_dataset(test_pct=0.1, img_size=(128,128), c=1, cond=True):
    
    dataset = DatasetLoader(size = img_size, c = c, cond = False)
    y_dataset = DatasetLoader(
        size = img_size,
        c=c,
        cond=True,
    )
    dataset = SequenceWrapper(dataset)
    y_dataset = SequenceWrapper(y_dataset)
    dataset = dataset.zip(y_dataset)
    test_size = int(len(dataset) * test_pct)
    train_size = int(len(dataset) - test_size)
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    return train_ds, test_ds