"""Programming Computer Vision with Python"""

# include <boost/whatever.hpp>

import os
from DICOM_test import dicom_read_and_write
import cv2
import sys
import numpy as np
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
import pysift
import sift

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
    conv_hull = convex_hull_image(openhull)
    # set of pixels included in smallest convex polygon that SURROUND all white pixels in the input image
    ch = np.multiply(conv_hull, 1)  # bool --> binary

    fore_image = ch * img  # phantom
    back_image = (1 - ch) * img  #background

    if slice_dim == 0:
        mask3D[imslice, :, :] = ch
    if slice_dim == 1:
        mask3D[:, imslice, :] = ch
    if slice_dim == 2:
        mask3D[:, :, imslice] = ch

idx = 0

for zz in np.linspace(7, 35, 29):  # slices 7-->36 are indexed 6-->35, 30 slices in total
    # actually want to start on index 7 so that zz-1 = 6 is slice 7
    print('Slice ', int(zz+1))
    zz = int(zz)  # slice of interest

    phmask1 = mask3D[zz-1, :, :]  # phantom mask
    phim1 = imdata[zz-1, :, :]*phmask1  # phantom image
    bgim1 = imdata[zz-1, :, :]*~phmask1  # background image

    phmask2 = mask3D[zz, :, :]  # phantom mask
    phim2 = imdata[zz, :, :] * phmask2  # phantom image
    bgim2 = imdata[zz, :, :] * ~phmask2  # background image

    phim_dims = np.shape(phim1)

    phim_norm1 = phim1/np.max(phim1)  # normalised image
    phim_gray1 = phim_norm1*255  # greyscale image
    grayinv1 = 255-phim_gray1

    phim_norm2 = phim2 / np.max(phim2)  # normalised image
    phim_gray2 = phim_norm2 * 255  # greyscale image
    grayinv2 = 255 - phim_gray2

    gray1 = grayinv1.copy()
    gray2 = grayinv2.copy()

    # display image
    cv2.imshow('inverted image 1', gray1.astype('uint8'))
    cv2.waitKey(0)
    cv2.imshow('inverted image 2', gray2.astype('uint8'))
    cv2.waitKey(0)

    kp1, dcp1 = pysift.computeKeypointsAndDescriptors(gray1, sigma=0.3, num_intervals=1, assumed_blur=0,
                                                      image_border_width=10)
    kp2, dcp2 = pysift.computeKeypointsAndDescriptors(gray2, sigma=0.3, num_intervals=1, assumed_blur=0,
                                                      image_border_width=10)

    im1 = phim1.copy()
    im2 = phim2.copy()

    img1 = phmask1.copy()/np.max(phmask1)
    for marker1 in kp1:
        j, k = tuple(int(i) for i in marker1.pt)
        if phmask1[j, k] == 1:  # if in phantom
            img1 = cv2.circle(img1, (j, k), 1, 0, 1)

    img2 = phmask2.copy() / np.max(phmask2)
    for marker2 in kp2:
        g, h = tuple(int(f) for f in marker2.pt)
        if phmask2[g, h] == 1:  # if in phantom
            img2 = cv2.circle(img2, (g, h), 1, 0, 1)

    cv2.imshow('Features Detected 1', img1)
    cv2.waitKey(0)

    cv2.imshow('Features Detected 2', img2)
    cv2.waitKey(0)

    scores_for_match = sift.match_twosided(dcp1, dcp2)

    sift.appendimages(im1, im2)

    sift.plot_matches(im1, im2, kp1, kp2, scores_for_match, phmask1+phmask2, show_below=True)



