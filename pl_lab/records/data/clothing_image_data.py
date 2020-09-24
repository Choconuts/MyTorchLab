import json
from PIL import Image


class ClothImageDataset:

    def __init__(self, cl_file, transform=None, target_transform=None, train=True):
        with open(cl_file, 'r') as fp:
            self.mapping = json.load(fp)
            import random
            random.seed(120)
            random.shuffle(self.mapping)
        if train:
            self.mapping = self.mapping[:-len(self.mapping) // 3]
        else:
            self.mapping = self.mapping[-len(self.mapping) // 3:]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image = Image.open(self.mapping[i][0])
        if self.transform is not None:
            image = self.transform(image)
        label = 0
        if self.mapping[i][1]:
            label = 1
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.mapping)