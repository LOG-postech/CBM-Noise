import pdb
import os
import sys
import copy
import ast
import random
import pickle
import argparse
import numpy as np
from pytorch_lightning import seed_everything

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.config import N_ATTRIBUTES, N_CLASSES

def add_concept_noise(pbm, out_dir, data_dir, noise_rate, n_concepts=85, n_classes=50, n_images=14928):
    """
    Add independent concept noise 
    """
    # Calculate the total number of attributes and the number of noisy attributes
    n_attrs = n_concepts * n_images
    n_noise = int(n_concepts * n_images * noise_rate)
    narray = np.concatenate((np.zeros(n_attrs - n_noise), np.ones(n_noise)))
    random.shuffle(narray)
    noise = narray.reshape((n_concepts, n_images))

    print("Concept noise array:", noise)
    print(noise.shape)

    def noise_fn(concepts, img_index, i):
        new_concepts = [int(not elem) if noise[j, i] else elem
                    for j, elem in enumerate(concepts)]
        return new_concepts, img_index

    create_new_dataset(pbm, out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def add_class_noise(pbm, out_dir, data_dir, noise_rate, n_class=50, n_images=14928):
    """
    Add independent class noise 
    """
    n_noise = int(n_images * noise_rate)
    noise = np.concatenate((np.zeros(n_images - n_noise), np.ones(n_noise)))
    random.shuffle(noise)

    print("Class noise array:", noise)
    print(noise.shape)

    def noise_fn(concepts, img_index, i):
        new_label = img_index
        if noise[i]:
            while new_label == img_index:
                new_label = random.randrange(0, n_class)
        return concepts, new_label

    create_new_dataset(pbm, out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def add_both_noise(pbm, out_dir, data_dir, noise_rate, n_concepts=85, n_class=50, n_images=14928):
    """
    Add independent both (concept and class) noise
    """
    # Calculate the total number of attributes and the number of noisy attributes
    n_attrs = n_concepts * n_images
    n_concept_noise = int(n_concepts * n_images * noise_rate)
    n_class_noise = int(n_images * noise_rate)

    # Generate concept noise
    concept_noise_array = np.concatenate((np.zeros(n_attrs - n_concept_noise), np.ones(n_concept_noise)))
    random.shuffle(concept_noise_array)
    concept_noise = concept_noise_array.reshape((n_concepts, n_images))

    # Generate class noise
    class_noise_array = np.concatenate((np.zeros(n_images - n_class_noise), np.ones(n_class_noise)))
    random.shuffle(class_noise_array)
    class_noise = class_noise_array

    print("Concept noise:", concept_noise)
    print(concept_noise.shape)
    print("Class noise:", class_noise)
    print(class_noise.shape)

    def noise_fn(concepts, img_index, i):
        # Apply concept noise
        new_concepts = [int(not elem) if concept_noise[j, i] else elem
                    for j, elem in enumerate(concepts)]

        # Apply class noise
        new_label = img_index
        if class_noise[i]:
            new_label = img_index
            while new_label == img_index:
                new_label = random.randrange(0, n_class)
        
        return new_concepts, new_label

    create_new_dataset(pbm, out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def create_new_dataset(pbm, out_dir, compute_fn, datasets=['train'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/add one field of the metadata in each dataset based on compute_fn
    and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
    and return the updated metadata object
    """    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.csv')
        if not os.path.exists(path):
            continue

        import pdb, pandas as pd

        df = pd.read_csv(path)

        new_img_names = []
        new_img_index = []
        new_concepts = []
        attribute_changed = 0
        class_changed = 0

        for index, row in df.iterrows(): 
            original_attribute = ast.literal_eval(row['concept'])
            original_class = row['img_index']

            rconcepts, rindex = compute_fn(original_attribute, original_class, index)

            new_img_names.append(row['img_name'])
            new_img_index.append(rindex)
            new_concepts.append(rconcepts)

            if original_class != rindex:
                class_changed += 1

            for j in range(len(original_attribute)):
                if original_attribute[j] != rconcepts[j]:
                    attribute_changed += 1
        
        print("[Statistics]")
        print("Dataset size : ", len(df))
        print("Attribute changed : ", attribute_changed, " / ", attribute_changed / (len(df) * 85))
        print("Class changed : ", class_changed, " / ", class_changed / len(df))

        new_df = pd.DataFrame({'img_name': new_img_names, 'img_index': new_img_index, 'concept': new_concepts})
        pd.DataFrame.to_csv(new_df, os.path.join(out_dir, dataset + '.csv'))


def get_data_len(data_dir=''):
    path = os.path.join(data_dir, 'train.pkl')
    if not os.path.exists(path):
        exit()
    data = pickle.load(open(path, 'rb'))
    return len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Noise type, choices : concept, class, asymConcept, ReduceMajorityVoting')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--image_dir', type=str, help='Image directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--noise_rate', type=float, help='noise rate')
    parser.add_argument('--n_concepts', type=int, help='number of concepts')
    args =  parser.parse_args()
    print(args)
    pbm = np.array(np.genfromtxt(f'{args.image_dir}/predicate-matrix-binary.txt', dtype='int'))
    seed_everything(23)

    if args.exp == 'concept' : 
        add_concept_noise(pbm, args.out_dir, args.data_dir, args.noise_rate)
    elif args.exp == 'class':
        add_class_noise(pbm, args.out_dir, args.data_dir, args.noise_rate)
    elif args.exp == 'both':
        add_both_noise(pbm, args.out_dir, args.data_dir, args.noise_rate)