"""
Create variants of the initial CUB dataset
"""
import pdb
import os
import sys
import copy
import torch
import random
import pickle
import argparse
import numpy as np
from PIL import Image
from shutil import copyfile
import torchvision.transforms as transforms
from collections import defaultdict as ddict
from pytorch_lightning import seed_everything

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def create_logits_data(model_path, out_dir, data_dir='', use_relu=False, use_sigmoid=False, backbone='inception_v3'):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path).cuda()

    get_logits_train = lambda d: inference(d['img_name'], model, use_relu, use_sigmoid, is_train=True, backbone=backbone)

    create_new_dataset(out_dir, get_logits_train, datasets=['train'], data_dir=data_dir)

def inference(img_path, model, use_relu, use_sigmoid, is_train, resol=299, layer_idx=None, backbone='inception_v3'):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()
    # see utils.py
    if 'vit_b16' in backbone or 'vit_l16' in backbone:
        if is_train:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])

    else:
        if is_train:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    input_var = torch.autograd.Variable(img_tensor).cuda()
    if layer_idx is not None:
        all_mods = list(model.modules())
        cropped_model = torch.nn.Sequential(*list(model.children())[:layer_idx])  # nn.ModuleList(all_mods[:layer_idx])
        print(type(input_var), input_var.shape, input_var)
        return cropped_model(input_var)

    outputs = model(input_var)
    if use_relu:
        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
    elif use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())

def create_new_dataset(out_dir, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.csv')
        if not os.path.exists(path):
            continue

        import ast, pandas as pd

        df = pd.read_csv(path)

        new_img_names = []
        new_img_index = []
        new_concepts = []

        for index, row in df.iterrows(): 
            new_concept_row = compute_fn(row)

            new_img_names.append(row['img_name'])
            new_img_index.append(row['img_index'])
            new_concepts.append(new_concept_row)

        new_df = pd.DataFrame({'img_name': new_img_names, 'img_index': new_img_index, 'concept': new_concepts})
        pd.DataFrame.to_csv(new_df, os.path.join(out_dir, dataset + '.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['ExtractConcepts', 'ExtractProbeRepresentations', 'DataEfficiencySplits', 'ChangeAdversarialDataDir', 'MajorityVoting', 'Reduced'],
                        help='Name of experiment to run.')
    parser.add_argument('--model_path', type=str, help='Path of model')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--adv_data_dir', type=str, help='Adversarial data directory')
    parser.add_argument('--train_splits', type=str, nargs='+', help='Train splits to use')
    parser.add_argument('--use_relu', action='store_true', help='Use Relu')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid')
    parser.add_argument('--layer_idx', type=int, default=None, help='Layer id to extract probe representations')
    parser.add_argument('--n_samples', type=int, help='Number of samples for data efficiency split')
    parser.add_argument('--splits_dir', type=str, help='Data dir of splits')
    parser.add_argument('--gpu_loc_data', type=str, help='GPU location data')
    parser.add_argument('--backbone', type=str, help='Backbone of the model')
    args = parser.parse_args()
    seed_everything(23)
    DATA_DIR = args.gpu_loc_data

    if args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_relu, args.use_sigmoid, args.backbone)
    else:
        exit()
