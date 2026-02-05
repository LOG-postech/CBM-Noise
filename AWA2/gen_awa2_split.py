import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pdb
from pytorch_lightning import seed_everything

CONCEPT_SEMANTICS = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']

def split_data(data_root, save_dir, val_split=0.2, test_split=0.4, train_split=0.4):
    predicate_binary_mat = np.array(np.genfromtxt(f'{data_root}/predicate-matrix-binary.txt', dtype='int')).tolist()

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(f"{data_root}/classes.txt") as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    class_to_index = class_to_index
    
    img_names = []
    img_index = []
    concepts = []
    for c in class_to_index.keys():
        class_name = c
        FOLDER_DIR = os.path.join(f'{data_root}/JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob.glob(file_descriptor)

        class_index = class_to_index[class_name]
        for file_name in files:
            img_names.append(file_name)
            img_index.append(class_index)
            concepts.append(predicate_binary_mat[class_index])

    img_names = img_names
    img_index = img_index
    
    # Split data into train and test
    train_img_names, val_img_names, train_img_index, val_img_index, train_concepts, val_concepts = train_test_split(img_names, img_index, concepts, train_size=train_split, random_state=42)
    val_img_names, test_img_names, val_img_index, test_img_index, val_concepts, test_concepts = train_test_split(val_img_names, val_img_index, val_concepts, train_size=val_split / (1. - train_split), random_state=42)
    
    train_df = pd.DataFrame({'img_name': train_img_names, 'img_index': train_img_index, 'concept': train_concepts})
    val_df = pd.DataFrame({'img_name': val_img_names, 'img_index': val_img_index, 'concept': val_concepts})
    test_df = pd.DataFrame({'img_name': test_img_names, 'img_index': test_img_index, 'concept': test_concepts})
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train: {len(train_df)/len(img_names)}, Val: {len(val_df)/len(img_names)}, Test: {len(test_df)/len(img_names)}")
    pd.DataFrame.to_csv(train_df, f'{save_dir}/train.csv')
    pd.DataFrame.to_csv(val_df, f'{save_dir}/val.csv')
    pd.DataFrame.to_csv(test_df, f'{save_dir}/test.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    args = parser.parse_args()
    seed_everything(23)
    split_data(args.data_dir, args.save_dir)