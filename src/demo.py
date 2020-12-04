import pdb
import cv2
import sift
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-l', default='lookup/main_sift.pkl', help='Path of the pre computed descriptor file')
parser.add_argument('-q', default='images/query/New_Belgium_Glutiny_Bottle.jpeg', help='Path of the query image')
args = parser.parse_args()

GOOD_MATCH_THRESH = 10

descriptor_file = args.l
query_image = args.q
matches_track = None
database_image = None
distances_track = list()

with open(descriptor_file, 'rb') as file:
    database_descriptors = pickle.load(file)

img_q = cv2.imread(query_image)
img_q_gray = cv2.imread(query_image,0)

# using the SIFT algorithm to generate keypoints and descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img_q_gray, verbose=True)
max_count = 0
for database_name in database_descriptors.keys():
    d_desc = database_descriptors[database_name]
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, d_desc, k=2)
    distances = list()
    counts = list()
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            distances.append(m)
    counts.append(len(distances))
    if len(distances) != 0:
        if np.max(counts) > max_count:
            max_count = np.max(counts)
            max_count_name = database_name.split('/')[-1].split('.')[0]
            matches_track = matches
            database_image = database_name.split('/')[-1]
            distances_track = distances
print("[INFO] Label for {}'s query image is: {}".format(query_image.split('/')[-1].split('.')[0], max_count_name))

print('\n[INFO] Preparing the visualization...\n')
extension_list = ['.jpeg', '.png', '.jpg', '.JPEG', '.PNG', '.JPG']
for extension in extension_list:
    img_d = cv2.imread('images/database/{}{}'.format(database_image, extension), 0)    
    if img_d is not None:
        break

keypoints_2, _ = sift.detectAndCompute(img_d, verbose=True)
try:
    matches = sorted(distances_track, key = lambda x:x.distance)
    matched_image = cv2.drawMatches(img_q_gray, keypoints_1, img_d, keypoints_2, matches, None, flags=2)
except:
    pdb.set_trace()
plt.imshow(matched_image)
plt.show()
