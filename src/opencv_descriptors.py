"""
This file contains code for generating features using OpenCV implementations for SIFT, SURF, and ORB
"""
import os
import cv2
import pdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--images', default='/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/images/database/', help='Path to the folder containing database images')
parser.add_argument('--descriptors', default='/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/src/lookups/', help='Path to the pickle file containing descriptors for database images')
parser.add_argument('--descpt_id', default='sift', help='Name of the descriptor to use. Options are: [SIFT, SURF, ORB]')
args = parser.parse_args()
print('[INFO] {}'.format(args))

args.descpt_id = args.descpt_id.lower()
assert args.descpt_id in ['surf', 'sift', 'orb'], "Invalid descriptor id. Valid options are: [SIFT, SURF, ORB]"
args.descriptors = os.path.join(args.descriptors, 'database_{}.pkl'.format(args.descpt_id))

images = os.listdir(args.images)
images = [os.path.join(args.images, item) for item in images]

if args.descpt_id == 'sift':
    feature_extractor = cv2.xfeatures2d.SIFT_create()
elif args.descpt_id == 'surf':
    feature_extractor = cv2.xfeatures2d.SURF_create(extended=True)   # For getting descriptor of size 128 instead of 64
else:
    feature_extractor = cv2.ORB_create()

if not os.path.exists(args.descriptors):
    print('[INFO] Creating descriptors...')
    descriptors = dict()
    for image in tqdm(images, desc='Processing images: '):
        image_gray = cv2.imread(image, 0)
        image_gray = cv2.resize(image_gray,(400,300))
        try:
            keypoint, descriptor = feature_extractor.detectAndCompute(image_gray, None)
        except Exception as e:
            print(e)
            pdb.set_trace()
        image_name = image.split('/')[-1].split('.')[0]
        if image_name not in descriptors.keys():
            descriptors[image_name] = descriptor
        else:
            print('[ERROR] Issue with file {}!\nCheck for duplicate images.'.format(image))
    with open(args.descriptors, 'wb') as file:
        pickle.dump(descriptors, file)
else:
    print('[INFO] Using previously saved descriptors...')
    with open(args.descriptors, 'rb') as file:
        descriptors = pickle.load(file)
    if len(descriptors.keys()) != len(images):
        raise ValueError("Number of descriptors do not match with number of images in the dataset folder. Create a new set of descriptors.")
