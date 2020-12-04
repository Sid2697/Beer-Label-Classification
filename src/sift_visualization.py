import cv2 
import matplotlib.pyplot as plt
import sift
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', default='images/samuel_adams/database/sam_adams_cream_stout.png', help='Path to the database image')
parser.add_argument('-q', default='images/samuel_adams/query/sam_adams_cream_stout.png', help='Path to the query image')
args = parser.parse_args()

GOOD_MATCH_THRESH = 10

query_images = args.q
database_images = args.d

#reading image
img_d = cv2.imread(database_images)  
img_q = cv2.imread(query_images)
img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)

#resizing both images to same shape
h1, w1 = img_d.shape[:2]
h2, w2 = img_q.shape[:2]
hdif = 0
if h1<h2:
    hdif = abs(int((h2 - h1) // 2))
    img_d = cv2.copyMakeBorder( img_d, hdif, hdif, 0, 0, cv2.BORDER_CONSTANT)
elif h1>h2:
    hdif = abs(int((h2 - h1) // 2))
    img_q = cv2.copyMakeBorder( img_q, hdif, hdif, 0, 0, cv2.BORDER_CONSTANT)        

img_d_gray = cv2.imread(database_images,0)  
img_q_gray = cv2.imread(query_images,0)

#use sift to get keypoints and descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img_q_gray, verbose=True)
keypoints_2, descriptors_2 = sift.detectAndCompute(img_d_gray, verbose=True)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
distances = []
# Using the ratio test provided by Lowe in the SIFT paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        distances.append(m)

matches = sorted(distances, key = lambda x:x.distance)
matched_image = cv2.drawMatches(img_q_gray, keypoints_1, img_d_gray, keypoints_2, matches, None, flags=2)
plt.imshow(matched_image)
plt.show()
# cv2.imwrite("output.jpg",cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR))
