"""
In this file we implement the SIFT algorithm.
This implementation closely follows OpenCV's implementation
References:
1. https://en.wikipedia.org/wiki/Trilinear_interpolation
2. https://www.wikiwand.com/en/Gaussian_blur
3. https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5
4. https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key


def get_base_image(image, sigma, initial_assumed_blur, verbose=True):
    """
    This method doubles the input image in size and applies Gaussian blur.
    We need to double the size of the input image by `diff` to achieve the blur of `sigma`.
    Source: https://www.wikiwand.com/en/Gaussian_blur
    """
    if verbose:
        print('[INFO] Generating the base image')
    updated_image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    diff = np.sqrt(max((sigma ** 2) - ((2 * initial_assumed_blur) ** 2), 0.01))
    base_image = cv2.GaussianBlur(updated_image, (0, 0), sigmaX=diff, sigmaY=diff)
    return base_image

def get_nos_octaves(shape):
    """
    This method is used for computing number of octaves in the image pyramid
    This is used to know the number of times we can half an image till it becomes too small.
    The `-1` in the equation ensures that we have the a side length of at least 3.
    """
    min_side = min(shape)
    octaves = round(np.log(min_side) / np.log(2) - 1)
    return int(octaves)

def get_Gaussian_kernels(sigma, num_intervals, verbose=True):
    """
    This method generates a list of kernels at which we wish to blur the input image.
    We use the value of sigma, intervals, and octaves from Section 3 of Lowe's paper.
    Interested users can refer to Fig. 3, 4, and 5 in the paper.
    """
    if verbose:
        print('[INFO] Getting scales')
    count_per_octave = num_intervals + 3    # Covering the images one step above the first image in the layer and one step below the last image in the layer
    const = 2 ** (1. / num_intervals)
    kernels = np.zeros(count_per_octave)
    kernels[0] = sigma

    for i in range(1, count_per_octave):
        prev_sigma = (const ** (i - 1)) * sigma
        total_sigma = const * prev_sigma
        kernels[i] = np.sqrt(total_sigma ** 2 - prev_sigma ** 2)
    return kernels

def get_Gaussian_images(image, octaves, kernels, verbose=True):
    """
    This method is used for generating a scale-space pyramid of Gaussian images
    """
    if verbose:
        print('[INFO] Preparing Gaussian images')
    images = list()
    for _ in range(octaves):
        images_in_octave = [image]
        for kernel in kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            images_in_octave.append(image)
        images.append(images_in_octave)
        base = images_in_octave[-3]
        image = cv2.resize(base, (int(base.shape[1]/2), int(base.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
    return np.array(images, dtype=object)

def get_DoG_images(gaus_images, verbose=True):
    """
    This method is used for generating the Difference of Gaussian (DoG) Pyramid
    """
    if verbose:
        print('[INFO] Preparing Difference of Gaussian images')
    images = list()
    for octave_images in gaus_images:
        images_octave = list()
        for orig_img, follow_img in zip(octave_images, octave_images[1:]):
            images_octave.append(np.subtract(follow_img, orig_img))
        images.append(images_octave)
    return np.array(images, dtype=object)

def get_scale_space_keypoints(gaus_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04, verbose=True):
    """
    This method is used to generate the keypoints using the Gaussian blurred and Laplacian images
    Here we are checking the neighbouring pixels for classifying the current pixel as extrema or not
    """
    if verbose:
        print('[INFO] Getting extrema in scale space')
    thresh = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # Using OpenCV's approach
    keypoints = list()
    for oct_i, dog_imgs_oct in enumerate(dog_images):
        for img_i, (first, second, third) in enumerate(zip(dog_imgs_oct, dog_imgs_oct[1:], dog_imgs_oct[2:])):
            for i in range(image_border_width, first.shape[0] - image_border_width):
                for j in range(image_border_width, first.shape[1] - image_border_width):
                    if is_extremum(first[i-1:i+2, j-1:j+2], second[i-1:i+2, j-1:j+2], third[i-1:i+2, j-1:j+2], thresh):
                        localization = localize_quadratic_fit(i, j, img_i + 1, oct_i, num_intervals, dog_imgs_oct, sigma, contrast_threshold, image_border_width)
                        if localization is not None:
                            keypoint, localized_i = localization
                            keypoints_oriented = get_keypoint_orientation(keypoint, oct_i, gaus_images[oct_i][localized_i])
                            for keypoint_oriented in keypoints_oriented:
                                keypoints.append(keypoint_oriented)
    return keypoints

def is_extremum(first_section, second_section, third_section, thresh):
    """
    This method determines if the point under consideration is an extrema or not for the given neighborhood
    """
    main_value = second_section[1, 1]
    if abs(main_value) > thresh:
        if main_value > 0:
            result = main_value >= second_section[1, 2] and np.all(main_value >= third_section) and np.all(main_value >= second_section[0, :]) and np.all(main_value >= second_section[2, :]) and main_value >= second_section[1, 0] and np.all(main_value >= first_section)
            return result
        elif main_value < 0:
            result = main_value <= second_section[1, 2] and np.all(main_value <= third_section) and np.all(main_value <= second_section[0, :]) and np.all(main_value <= second_section[2, :]) and main_value <= second_section[1, 0] and np.all(main_value <= first_section)
            return result
    return False

def localize_quadratic_fit(i, j, img_i, oct_i, num_intervals, dog_imgs_oct, sigma, contrast_threshold, image_border_width, eigen_ratio=10, num_attempts=5, verbose=False):
    """
    This method is used for fitting a quadratic curve to the extrema obtained using the state space.
    This is an iterative method and uses the extrema's neighbours for fitting the curve.
    This helps in refining and improving the position of the extrema.
    """
    if verbose:
        print('[INFO] Fitting the localization curve on state-space extrema')
    extrema_outside = False
    for attempt in range(num_attempts):
        first, second, third = dog_imgs_oct[img_i-1:img_i+2]
        cube = np.stack([first[i-1:i+2, j-1:j+2], second[i-1:i+2, j-1:j+2], third[i-1:i+2, j-1:j+2]]).astype('float32')/255. # rescaling pixel values to [0, 1] in the neighbourhood under consideration
        grad = get_grad(cube)
        hess = get_hess(cube)
        new_extremum = -np.linalg.lstsq(hess, grad, rcond=None)[0]
        if abs(new_extremum[0]) < 0.5 and abs(new_extremum[1]) < 0.5 and abs(new_extremum[2]) < 0.5:
            break
        i += int(round(new_extremum[1]))
        j += int(round(new_extremum[0]))
        img_i += int(round(new_extremum[2]))
        if i < image_border_width or i >= dog_imgs_oct[0].shape[0] - image_border_width or j < image_border_width or j >= dog_imgs_oct[0].shape[1] - image_border_width or img_i < 1 or img_i > num_intervals:
            extrema_outside = True
            break
    if extrema_outside:
        return None
    if attempt >= num_attempts - 1:
        return None
    new_extremum_value = cube[1, 1, 1] + 0.5 * np.dot(grad, new_extremum)
    if abs(new_extremum_value) * num_intervals >= contrast_threshold:
        xy_hess = hess[:2, :2]
        xy_hess_trace = np.trace(xy_hess)
        xy_hess_det = np.linalg.det(xy_hess)
        if xy_hess_det > 0 and eigen_ratio * (xy_hess_trace ** 2) < ((eigen_ratio + 1)**2) * xy_hess_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + new_extremum[0]) * (2 ** oct_i), (i + new_extremum[1]) * (2 ** oct_i))
            keypoint.octave = oct_i + img_i * (2 ** 8) + int(round((new_extremum[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((img_i + new_extremum[2]) / np.float32(num_intervals))) * (2 ** (oct_i + 1))
            keypoint.response = abs(new_extremum_value)
            return keypoint, img_i
    return None

def get_grad(cube):
    ds = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
    dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
    dy = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
    return np.array([dx, dy, ds])

def get_hess(cube):
    main_val = cube[1, 1, 1]
    dss = cube[2, 1, 1] - 2 * main_val + cube[0, 1, 1]
    dxx = cube[1, 1, 2] - 2 * main_val + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2 * main_val + cube[1, 0, 1]
    dxs = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
    dys = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
    dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
    hess = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    return hess

def get_keypoint_orientation(keypoint, oct_i, gaus_image, radius_factor=3, bins=36, peak_r=0.8, s_factor=1.5, verbose=False):
    """
    This method is used for calculating the orientation for each and every keypoint
    """
    if verbose:
        print('[INFO] Calculating orientations for keypoints')
    keypoint_orientations = []
    scale = s_factor * keypoint.size / np.float32(2 ** (oct_i + 1))
    rad = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    hist_raw = np.zeros(bins)
    hist_smooth = np.zeros(bins)

    for i in range(-rad, rad + 1):
        y_reg = int(round(keypoint.pt[1] / np.float32(2 ** oct_i))) + i
        if y_reg > 0 and y_reg < gaus_image.shape[0] - 1:
            for j in range(-rad, rad + 1):
                x_reg = int(round(keypoint.pt[0] / np.float32(2 ** oct_i))) + j
                if x_reg > 0 and x_reg < gaus_image.shape[1] - 1:
                    dx = gaus_image[y_reg, x_reg + 1] - gaus_image[y_reg, x_reg - 1]
                    dy = gaus_image[y_reg - 1, x_reg] - gaus_image[y_reg + 1, x_reg]
                    magnitude = np.sqrt(dx * dx + dy * dy)
                    orient = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    index = int(round(orient * bins / 360.))
                    hist_raw[index % bins] += weight * magnitude
    for n in range(bins):
        hist_smooth[n] = (6 * hist_raw[n] + 4 * (hist_raw[n - 1] + hist_raw[(n + 1) % bins]) + hist_raw[n - 2] + hist_raw[(n + 2) % bins]) / 16.
    max_orient = max(hist_smooth)
    peaks_orient = np.where(np.logical_and(hist_smooth > np.roll(hist_smooth, 1), hist_smooth > np.roll(hist_smooth, -1)))[0]
    for peak_i in peaks_orient:
        peak_v = hist_smooth[peak_i]
        """
        Interpolation update referred from equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        """
        if peak_v >= peak_r * max_orient:
            left = hist_smooth[(peak_i - 1) % bins]
            right = hist_smooth[(peak_i + 1) % bins]
            interpolated_peak_i = (peak_i + 0.5 * (left - right) / (left - 2 * peak_v + right)) % bins
            orient = 360. - interpolated_peak_i * 360. / bins
            if abs(orient - 360.) < 1e-7:
                orient = 0
            updated_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orient, keypoint.response, keypoint.octave)
            keypoint_orientations.append(updated_keypoint)
    return keypoint_orientations

def compare_kp(kp1, kp2):
    """
    This method returns True is kp1 is less than kp2
    """
    if kp1.pt[0] != kp2.pt[0]:
        return kp1.pt[0] - kp2.pt[0]
    if kp1.pt[1] != kp2.pt[1]:
        return kp1.pt[1] - kp2.pt[1]
    if kp1.size != kp2.size:
        return kp2.size - kp1.size
    if kp1.angle != kp2.angle:
        return kp1.angle - kp2.angle
    if kp1.response != kp2.response:
        return kp2.response - kp1.response
    if kp1.octave != kp2.octave:
        return kp2.octave - kp1.octave
    return kp2.class_id - kp1.class_id

def cvt_input_img_size(keypoints):
    """
    This method is used to convert properties of keypoint to the input image's size
    """
    updated_kp = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        updated_kp.append(keypoint)
    return updated_kp

def remove_dup_and_cvt(keypoints):
    """
    This method is used for sorting keypoints and removing the duplicates
    """
    if len(keypoints) < 2:
        return keypoints
    keypoints.sort(key=cmp_to_key(compare_kp))
    unique_kp = [keypoints[0]]
    for next_kp in keypoints[1:]:
        unq_last_kp = unique_kp[-1]
        if unq_last_kp.pt[0] != next_kp.pt[0] or unq_last_kp.pt[1] != next_kp.pt[1] or unq_last_kp.size != next_kp.size or unq_last_kp.angle != next_kp.angle:
            unique_kp.append(next_kp)
    return cvt_input_img_size(unique_kp)

def get_descriptors(keypoints, gaus_images, width=4, bins_count=8, scale_times=3, desc_max=0.2, verbose=True):
    """
    This method is used for generating the descriptors for each keypoint.
    """
    if verbose:
        print('[INFO] Generating descriptors')
    desc = []
    for keypoint in keypoints:
        row_bin_track = []
        col_bin_track = []
        mag_track = []
        orien_bin_track = []
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        if octave >= 0:
            scale = 1 / np.float32(1 << octave)
        else:
            scale = np.float32(1 << -octave)
        gaus_image = gaus_images[octave + 1, layer]
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_degree = bins_count / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_mul = -0.5 / ((0.5 * width) ** 2)
        histogram = np.zeros((width + 2, width + 2, bins_count))
        histogram_width = scale_times * 0.5 * scale * keypoint.size
        histogram_half_width = int(np.round(histogram_width * np.sqrt(2) * (width + 1) * 0.5))
        histogram_half_width = int(min(histogram_half_width, np.sqrt(gaus_image.shape[0] ** 2 + gaus_image.shape[1] ** 2)))
        for row in range(-histogram_half_width, histogram_half_width + 1):
            for column in range(-histogram_half_width, histogram_half_width + 1):
                row_rotation = column * sin_angle + row * cos_angle
                column_rotation = column * cos_angle - row * sin_angle
                row_bin_loc = (row_rotation / histogram_width) + 0.5 * width - 0.5
                column_bin_loc = (column_rotation / histogram_width) + 0.5 * width - 0.5
                if row_bin_loc > -1 and row_bin_loc < width and column_bin_loc > -1 and column_bin_loc < width:
                    row_window = int(round(point[1] + row))
                    column_window = int(round(point[0] + column))
                    if row_window > 0 and row_window < gaus_image.shape[0] - 1 and column_window > 0 and column_window < gaus_image.shape[1] - 1:
                        dx = gaus_image[row_window, column_window + 1] - gaus_image[row_window, column_window - 1]
                        dy = gaus_image[row_window - 1, column_window] - gaus_image[row_window + 1, column_window]
                        grad_mag = np.sqrt(dx * dx + dy * dy)
                        grad_orient = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_mul * ((row_rotation / histogram_width) ** 2 + (column_rotation / histogram_width) ** 2))
                        row_bin_track.append(row_bin_loc)
                        col_bin_track.append(column_bin_loc)
                        mag_track.append(weight * grad_mag)
                        orien_bin_track.append((grad_orient - angle) * bins_degree)
        # Smoothing via trilinear interpolation https://en.wikipedia.org/wiki/Trilinear_interpolation
        for row_bin_loc, column_bin_loc, magnitude, orientation_bin in zip(row_bin_track, col_bin_track, mag_track, orien_bin_track):
            row_bin_floor, column_bin_floor, orient_bin_floor = np.floor([row_bin_loc, column_bin_loc, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin_loc - row_bin_floor, column_bin_loc - column_bin_floor, orientation_bin - orient_bin_floor
            if orient_bin_floor < 0:
                orient_bin_floor += bins_count
            if orient_bin_floor >= bins_count:
                orient_bin_floor -= bins_count
            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)
            histogram[row_bin_floor + 1, column_bin_floor + 1, orient_bin_floor] += c000
            histogram[row_bin_floor + 1, column_bin_floor + 1, (orient_bin_floor + 1) % bins_count] += c001
            histogram[row_bin_floor + 1, column_bin_floor + 2, orient_bin_floor] += c010
            histogram[row_bin_floor + 1, column_bin_floor + 2, (orient_bin_floor + 1) % bins_count] += c011
            histogram[row_bin_floor + 2, column_bin_floor + 1, orient_bin_floor] += c100
            histogram[row_bin_floor + 2, column_bin_floor + 1, (orient_bin_floor + 1) % bins_count] += c101
            histogram[row_bin_floor + 2, column_bin_floor + 2, orient_bin_floor] += c110
            histogram[row_bin_floor + 2, column_bin_floor + 2, (orient_bin_floor + 1) % bins_count] += c111
        desc_vect = histogram[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize desc_vect
        threshold = np.linalg.norm(desc_vect) * desc_max
        desc_vect[desc_vect > threshold] = threshold
        desc_vect /= max(np.linalg.norm(desc_vect), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        desc_vect = np.round(512 * desc_vect)
        desc_vect[desc_vect < 0] = 0
        desc_vect[desc_vect > 255] = 255
        desc.append(desc_vect)
    return np.array(desc, dtype='float32')

def visualize_pyramid(gaus_images, layer):
    """
    This method is used to visualise a layer from the image pyramid
    """
    images = gaus_images[layer]
    fig = plt.figure()
    if len(images) != 6 and len(images) != 5:
        print('[ERROR] Number of images != 6; they are equal to == {}'.format(len(images)))
        return None
    for i in range(len(images)):
        fig.add_subplot(2, 3, i + 1)
        plt.title('{} image'.format(i + 1))
        plt.imshow(images[i], 'gray')
        plt.xticks([]);plt.yticks([])
    plt.show()
    return None

def detectAndCompute(image, sigma=1.6, num_intervals=3, initial_assumed_blur=0.5, image_border_width=5, verbose=False):
    """
    This method is used for generating SIFT keypoints and detectors.
    All the values (sigma, num_intervals, initial_assumed_blur) are used from the paper by Lowe.
    """    
    image = image.astype('float32')
    base_image = get_base_image(image, sigma, initial_assumed_blur, verbose=verbose)
    octaves = get_nos_octaves(base_image.shape)
    kernels = get_Gaussian_kernels(sigma, num_intervals, verbose=verbose)
    gaus_images = get_Gaussian_images(base_image, octaves, kernels, verbose=verbose)
    dog_images = get_DoG_images(gaus_images, verbose=verbose)
    keypoints = get_scale_space_keypoints(gaus_images, dog_images, num_intervals, sigma, image_border_width, verbose=verbose)
    keypoints = remove_dup_and_cvt(keypoints)
    descriptors = get_descriptors(keypoints, gaus_images, verbose=verbose)
    return keypoints, descriptors
