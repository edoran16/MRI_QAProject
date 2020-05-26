"""Programming Computer Vision with Python"""

#include <boost/whatever.hpp>

import os
from PIL import Image
from DICOM_test import dicom_read_and_write
import cv2
import sys
import numpy as np
from skimage import filters
from skimage.morphology import convex_hull_image, convex_hull_object, opening
from skimage import exposure as ex
import pysift
import sift
import matplotlib.pyplot as plt
from skimage.measure import profile_line, label, regionprops


directpath = "MagNET_acceptance_test_data/scans/"
folder = "42-SLICE_POS"

pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

with os.scandir(pathtodicom) as it:
    for file in it:
        path = "{0}{1}".format(pathtodicom, file.name)

ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

try:
    xdim, ydim, zdim = dims
    #OrthoSlicer3D(imdata).show()  # look at 3D volume data
except ValueError:
    print('DATA INPUT ERROR: this is 2D image data')
    sys.exit()

# create 3D mask
slice_dim = np.where(dims == np.min(dims))
slice_dim = slice_dim[0]
slice_dim = slice_dim[0]
no_slices = dims[slice_dim]
print("Number of slices = ", no_slices)
mask3D = np.zeros_like(imdata)

for imslice in np.linspace(0, no_slices-1, no_slices, dtype=int):
    if slice_dim == 0:
        img = imdata[imslice, :, :]  #sagittal
    if slice_dim == 1:
        img = imdata[:, imslice, :]  # coronal
        # TODO: might need to "squeeze out" middle dimension?
    if slice_dim == 2:
        img = imdata[:, :, imslice]  # transverse

    h = ex.equalize_hist(img) * 255  # histogram equalisation increases contrast of image
    oi = np.zeros_like(img, dtype=np.uint16)  # creates zero array same dimensions as img
    oi[(img > filters.threshold_otsu(img)) == True] = 1  # Otsu threshold on image
    oh = np.zeros_like(img, dtype=np.uint16)  # zero array same dims as img
    oh[(h > filters.threshold_otsu(h)) == True] = 1  # Otsu threshold on hist eq image

    nm = img.shape[0] * img.shape[1]  # total number of voxels in image
    # calculate normalised weights for weighted combination
    w1 = np.sum(oi) / nm
    w2 = np.sum(oh) / nm
    ots = np.zeros_like(img, dtype=np.uint16)  # create final zero array
    new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
    ots[(new > filters.threshold_otsu(new)) == True] = 1  # Otsu threshold on weighted combination

    openhull = opening(ots)
    conv_hull = convex_hull_image(openhull)  # set of pixels included in smallest convex polygon that SURROUND all white pixels in the input image
    ch = np.multiply(conv_hull, 1)  # bool --> binary

    # cv2.imshow('otsu', (ots * 255).astype('uint8'))
    # cv2.waitKey(0)
    # cv2.imshow('opening on otsu', (openhull * 255).astype('uint8'))
    # cv2.waitKey(0)
    # cv2.imshow('convex hull on opening', (conv_hull * 255).astype('uint8'))
    # cv2.waitKey(0)
    # cv2.imshow('final mask', (ch * 255).astype('uint8'))
    # cv2.waitKey(0)

    fore_image = ch * img  # phantom
    back_image = (1 - ch) * img  #background

    if slice_dim == 0:
        mask3D[imslice, :, :] = ch
    if slice_dim == 1:
        mask3D[:, imslice, :] = ch
    if slice_dim == 2:
        mask3D[:, :, imslice] = ch

# For slice position analysis want to do analysis on every slice but for now start with mid-slice
# TODO: only interested in slices 7 to 36 as this is where rods are... need to detect this range!!
# TODO: make this code work for every slice! and make measurement

idx = 0

for zz in [22, 23]:#np.linspace():#(6, 35, 30):#range(no_slices):
    print('Slice ', int(zz+1))
    zz = int(zz)  # slice of interest
    phmask = mask3D[zz, :, :]  # phantom mask
    phim = imdata[zz, :, :]*phmask  # phantom image
    bgim = imdata[zz, :, :]*~phmask  # background image

    phim_dims = np.shape(phim)

    phim_norm = phim/np.max(phim)  # normalised image
    phim_gray = phim_norm*255  # greyscale image

    grayinv = 255-phim_gray

    gray = grayinv.copy()

    # display image
    cv2.imshow('inverted image', gray.astype('uint8'))
    cv2.waitKey(0)

    keypoints, descriptors = pysift.computeKeypointsAndDescriptors(gray, sigma=0.3, num_intervals=1, assumed_blur=0, image_border_width=10)

    if idx == 0:
        kp1 = keypoints
        dcp1 = descriptors
        im1 = phim
    if idx == 1:
        kp2 = keypoints
        dcp2 = descriptors
        im2 = phim

    idx = idx + 1

    img = phmask.copy()/np.max(phmask)
    count = 1
    for marker in keypoints:
        j, k = tuple(int(i) for i in marker.pt)
        if phmask[j, k] == 1:  # if in phantom
            img = cv2.circle(img, (j, k), 1, 0, 1)
            #print(count)
            count = count + 1

    cv2.imshow('Features Detected', img)
    cv2.waitKey(0)

print(np.shape(dcp1), np.shape(dcp2))

scores_for_match = sift.match_twosided(dcp1, dcp2)

print(np.shape(scores_for_match))

sift.appendimages(im1, im2)

sift.plot_matches(im1, im2, kp1, kp2, scores_for_match, phmask, show_below=True)


#cv2.imwrite('sift_keypoints.jpg', img)


