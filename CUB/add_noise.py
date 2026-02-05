import pdb
import os
import sys
import copy
import random
import pickle
import argparse
import numpy as np
from scipy.stats import norm
from pytorch_lightning import seed_everything

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.config import N_ATTRIBUTES, N_CLASSES

def add_concept_noise(out_dir, data_dir, noise_rate, n_concepts=112, n_classes=200, n_images=11788):
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

    def noise_fn(d, i):
        d['attribute_label'] = [int(not elem) if noise[j, i] else elem
                                for j, elem in enumerate(d['attribute_label'])]
        return d

    create_new_dataset(out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def add_class_noise(out_dir, data_dir, noise_rate, n_class=200, n_images=11788):
    """
    Add independent class noise 
    """
    n_noise = int(n_images * noise_rate)
    noise = np.concatenate((np.zeros(n_images - n_noise), np.ones(n_noise)))
    random.shuffle(noise)

    print("Class noise array:", noise)
    print(noise.shape)

    def noise_fn(d, i):
        if noise[i]:
            original_label = d['class_label']
            new_label = original_label
            while new_label == original_label:
                new_label = random.randrange(0, n_class)
            d['class_label'] = new_label
        return d

    create_new_dataset(out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def add_both_noise(out_dir, data_dir, noise_rate, n_concepts=112, n_class=200, n_images=11788):
    """
    Add independent concept & class noise
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

    def noise_fn(d, i):
        # Apply concept noise
        d['attribute_label'] = [int(not elem) if concept_noise[j, i] else elem
                                for j, elem in enumerate(d['attribute_label'])]

        # Apply class noise
        if class_noise[i]:
            original_label = d['class_label']
            new_label = original_label
            while new_label == original_label:
                new_label = random.randrange(0, n_class)
            d['class_label'] = new_label
        
        return d

    create_new_dataset(out_dir, noise_fn, datasets=['train'], data_dir=data_dir)


def create_new_dataset(out_dir, compute_fn, datasets=['train'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/add one field of the metadata in each dataset based on compute_fn
    and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
    and return the updated metadata object
    """    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        
        data = pickle.load(open(path, 'rb'))
        new_data = []

        attribute_changed = 0
        class_changed = 0
        
        for i, d in enumerate(data):
            original_attribute = d['attribute_label']
            original_class = d['class_label']

            new_d = copy.deepcopy(d)
            new_d = compute_fn(new_d, i)
            new_data.append(new_d)

            if original_class != new_d['class_label']:
                class_changed += 1

            for j in range(len(original_attribute)):
                if original_attribute[j] != new_d['attribute_label'][j]:
                    attribute_changed += 1
        
        print("[Statistics]")
        print("Dataset size : ", len(data))
        print("Attribute changed : ", attribute_changed, " / ", attribute_changed / (len(data) * len(data[0]['attribute_label'])))
        print("Class changed : ", class_changed, " / ", class_changed / len(data))
        
        with open(os.path.join(out_dir, dataset + '.pkl'), 'wb') as f:
            pickle.dump(new_data, f)


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
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--noise_rate', type=float, help='noise rate')
    parser.add_argument('--n_concepts', type=int, help='number of concepts')
    args =  parser.parse_args()
    print(args)
    dataset_size = get_data_len(args.data_dir)
    seed_everything(23)

    if args.exp == 'concept' : 
        add_concept_noise(args.out_dir, args.data_dir, args.noise_rate, n_images=dataset_size)
    elif args.exp == 'class':
        add_class_noise(args.out_dir, args.data_dir, args.noise_rate, n_images=dataset_size)
    elif args.exp == 'both':
        add_both_noise(args.out_dir, args.data_dir, args.noise_rate, n_images=dataset_size)