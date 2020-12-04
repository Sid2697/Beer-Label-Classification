import cv2
import sift
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', default='images/database/', help='Path to the database images')
parser.add_argument('-q', default='images/query/', help='Path to the query image')
parser.add_argument('-l', default='lookup/main_sift.pkl', help='Path to the pre computed descriptors')
parser.add_argument('-load', default='Y', help='Load pre computed descriptors [Y/N]')
args = parser.parse_args()

GOOD_MATCH_THRESH = 10

query_images = args.q
database_images = args.d
descriptor_file = args.l
load_descriptor = args.load

database_descriptors = {}
name_list = []

if load_descriptor == 'Y' or load_descriptor =='y':
    print("Loading Pre Computed Descriptors")
    assert os.path.exists(descriptor_file)
    with open(descriptor_file, 'rb') as file:
        database_descriptors = pickle.load(file)
else:
    for image_name in tqdm(os.listdir(database_images)):
        img = cv2.imread(database_images+image_name,0)
        keypoint, descriptor = sift.detectAndCompute(img)
        img_name,img_extension = os.path.splitext(image_name)
        database_descriptors[img_name] = descriptor
        name_list.append(img_name)
    with open(descriptor_file, 'wb') as file:
        pickle.dump(database_descriptors, file)

print("--Database Descriptors Loaded--")

correct_count = 0


for query_name in tqdm(os.listdir(query_images)):
    max_count_name = None
    max_count = 0
    query = cv2.imread(query_images+query_name, 0)
    _, q_desc = sift.detectAndCompute(query)
    for database_name in database_descriptors.keys():
        d_desc = database_descriptors[database_name]
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(q_desc, d_desc, k=2)
        matches_mask = [[0, 0] for i in range(len(matches))]
        distances = list()
        counts = list()
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
                distances.append(m.distance)
        counts.append(len(distances))
        if len(distances) != 0:
            if np.max(counts) > max_count:
                max_count = np.max(counts)
                max_count_name = database_name.split('/')[-1].split('.')[0]
    if query_name.split('/')[-1].split('.')[0] == max_count_name:
        correct_count += 1
    else:
        print('[ERROR] database_image: {}, query_image: {}'.format(max_count_name, query_name.split('/')[-1].split('.')[0]))
print('[INFO] Correct matches: {} total queries: {} Accuracy: {}'.format(correct_count, len(os.listdir(query_images)), correct_count/len(os.listdir(query_images))))
