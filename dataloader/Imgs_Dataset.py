import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd


class Imgs_Dataset(Dataset):
    def __init__(self, args, list_file, mode, transforms):
        # Basic info
        self.args = args
        self.mode = mode
        self.transforms = transforms

        # File path
        label_df = pd.read_csv(list_file)
        
        # Imgs & Labels
        self.names               = label_df['filePath']
        self.cate_labels         = torch.from_numpy(np.array(label_df['expression']))

    def __len__(self):
        return len(self.cate_labels)

    def __getitem__(self, index):
        img = Image.open(self.names[index])
        img = self.transforms(img)
        cate_label = self.cate_labels[index]

        return img, cate_label


def train_data_loader(args):
    image_size = 224
    train_transforms = torchvision.transforms.Compose([transforms.RandomResizedCrop((image_size, image_size)),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor()])
    train_data = Imgs_Dataset(args=args,
                              list_file=args.list_file_train,
                              mode='train',
                              transforms=train_transforms)
    return train_data

def test_data_loader(args):
    image_size = 224
    test_transforms = torchvision.transforms.Compose([transforms.Resize((image_size, image_size)),
                                                      transforms.ToTensor()])
    test_data = Imgs_Dataset(args=args,
                             list_file=args.list_file_val,
                             mode='test',
                             transforms=test_transforms)
    return test_data


if __name__ == "__main__":
    dataloader_te = test_data_loader()



        

        




