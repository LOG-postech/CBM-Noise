import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import pdb
import ast
########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50
SElECTED_CONCEPTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
CONCEPT_SEMANTICS = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']

class AnimalDataset(Dataset):
  def __init__(self, args, image_dir, data_path, transform):
    root_dir=image_dir
    self.transform = transform
    self.args = args

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(f"{root_dir}/classes.txt") as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    df = pd.read_csv(data_path)
    img_names = df['img_name'].tolist()
    img_index = df['img_index'].tolist()
    concepts = df['concept'].tolist()
    for i in range(len(concepts)):
      concepts[i] = ast.literal_eval(concepts[i])
   
    self.img_names = img_names
    self.img_index = img_index
    self.concepts = concepts

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)

    im_index = self.img_index[index]
    im_concepts = self.concepts[index]

    if self.args.use_attr:
      if self.args.no_img:
        return torch.FloatTensor(im_concepts), im_index
      else:
        return im, im_index, torch.FloatTensor(im_concepts)
    else:
      return im, im_index

  def __len__(self):
    return len(self.img_names)


def load_data(args, image_dir, data_path, batch_size, shuffle=True, num_workers=4, resol=224, is_training=True):
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    dataset = AnimalDataset(args, image_dir, data_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, dataset


def generate_data(config, image_dir, train_dir, val_dir, test_dir, seed=42, output_dataset_vars=False, resol=224, is_training=True):
    concept_group_map = None
    n_concepts = len(SElECTED_CONCEPTS)
    seed_everything(seed)
    
    if test_dir == "":
      train_dl, dataset = load_data(config, image_dir, train_dir, config.batch_size, shuffle=True, num_workers=8, resol=resol, is_training=True)
      val_dl, _ = load_data(config, image_dir, val_dir, config.batch_size, shuffle=False, num_workers=8, resol=resol, is_training=False)
      test_dl = None
    else:
      train_dl = None
      val_dl = None
      test_dl, _ = load_data(config, image_dir, test_dir, config.batch_size, shuffle=False, num_workers=8, resol=resol, is_training=False)
      return train_dl, val_dl, test_dl, None, (n_concepts, N_CLASSES, concept_group_map)
    
    if config.weighted_loss == "multiple":
      imbalance = find_class_imbalance_awa2(train_dir, dataset, n_concepts, multiple_attr=True)
    else:
      imbalance = None

    return train_dl, val_dl, test_dl, imbalance, (n_concepts, N_CLASSES, concept_group_map)
    

def find_class_imbalance_awa2(csv_file, dataset, n_attr, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pd.read_csv(csv_file)
    n = len(data)
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]

    import ast
    for index, row in data.iterrows():
        concepts = ast.literal_eval(row['concept'])
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += concepts[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += concepts[attr_idx]
            else:
                n_ones[0] += sum(concepts)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr:
        imbalance_ratio *= n_attr
    return imbalance_ratio
