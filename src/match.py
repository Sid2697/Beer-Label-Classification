"""
Given features from two images, this file is used to match those features and find the matches between them
References: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""
import pdb
import cv2
import glob
import sift
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', default='images/database/*', help='Path to the database images')
parser.add_argument('-q', default='images/query/*', help='Path to the query image')
args = parser.parse_args()

GOOD_MATCH_THRESH = 10

query_images = glob.glob(args.q)
database_images = glob.glob(args.d)
feature_extractor = cv2.xfeatures2d.SIFT_create()
with open('src/lookups/database_sift.pkl', 'rb') as file:
    database_features = pickle.load(file)
correct_count = 0

feature_extractor = cv2.xfeatures2d.SIFT_create()

for query_name in query_images:
    max_count_name = None
    max_count = 0
    query = cv2.imread(query_name, 0)
    # _, q_desc = sift.main(query, verbose=True)
    _, q_desc = feature_extractor.detectAndCompute(query, None)
    for database_name in database_images:
        d_desc = database_features[database_name.split('/')[-1].split('.')[0]]
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
    # print('[INFO] Max count: {} Database Image name: {} Query image name: {}'.format(max_count, max_count_name, query_name))
    if query_name.split('/')[-1].split('.')[0] == max_count_name:
        correct_count += 1
    else:
        print('[ERROR] database_image: {}, query_image: {}'.format(max_count_name, query_name.split('/')[-1].split('.')[0]))
print('[INFO] Correct matches: {} total queries: {} Accuracy: {}'.format(correct_count, len(query_images), correct_count/len(query_images)))
