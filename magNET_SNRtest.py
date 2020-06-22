""" MagNET SNR measurements. Aim to get code to work on head, body and spine data """

# load SNR TRA NICL
# get code working - then extend to SAG and COR
# then extend to oil phantom
# then look at body coil

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import re
import pandas as pd
import snr_funcs as sf
from skimage.draw import ellipse
from scipy import ndimage

from scipy.signal import find_peaks, argrelmin
from DICOM_test import dicom_read_and_write
from skimage import filters
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/SNR_Images/"

show_graphical = False  # display image processing steps
show_quad = False  # show quadrants for determining signal ROIs on marker image
show_bbox = False  # show bounding box of phantom on marker image

test_object = ['FloodField', 'Spine', 'Shoulder']
phantom_type = ['NICL', 'OIL']
geos = ['_TRA_', '_SAG_', '_COR_']

# TODO: iterate through different datasets

# TODO: classes will be useful here I think.....

caseT = False
if caseT:
    geo = '_TRA_'

caseS = True
if caseS:
    geo = '_SAG_'

caseC = False
if caseC:
    geo = '_COR_'

with os.scandir(directpath) as the_folders:
    for folder in the_folders:
        fname = folder.name
        if re.search('-SNR_', fname):
            if re.search(geo, fname):
                if re.search('_NICL', fname):
                    if not re.search('_REPEAT', fname) and not re.search('_PR', fname):
                        print('Loading ', geo, 'geometry, with ', '_NICL_', 'phantom', 'for', 'FloodField', 'test object')
                        print('Loading ...', fname)
                        folder = fname
                        pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                        with os.scandir(pathtodicom) as it:
                            for file in it:
                                path = "{0}{1}".format(pathtodicom, file.name)

                        ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

try:
    xdim, ydim = dims
    print('Matrix Size =', xdim, 'x', ydim)

    # TODO: read in any useful metadata for SNR measures???

    img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

    cv2.imshow('dicom imdata', img)
    cv2.waitKey(0)

except ValueError:
    print('DATA INPUT ERROR: this is 3D image data')
    OrthoSlicer3D(imdata).show()  # look at 3D volume data
    sys.exit()

# mask phantom and background
mask = sf.create_2D_mask(img)  # boolean
bin_mask = (mask / np.max(mask)).astype('uint8')  # binary mask [0, 1]

if show_graphical:
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

# draw signal ROIs
# get centre of phantom and definte 5 ROIs from there
label_img, num = label(bin_mask, connectivity=imdata.ndim, return_num=True)  # labels the mask

props = regionprops(label_img)  # returns region properties for phantom mask ROI
phantom_centre = props[0].centroid
pc_row, pc_col = [int(phantom_centre[0]), int(phantom_centre[1])]
print(pc_row, pc_col)

# centre signal ROI
signal_roi_1 = img[pc_row, pc_col]

# show detected regions and lines on marker_im
marker_im = img.copy()
marker_im = marker_im.astype('uint8')
marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

cv2.line(marker_im, (pc_col + 10, pc_row + 10), (pc_col + 10, pc_row - 10), (0, 0, 255), 1)
cv2.line(marker_im, (pc_col + 10, pc_row - 10), (pc_col - 10, pc_row - 10), (0, 0, 255), 1)
cv2.line(marker_im, (pc_col - 10, pc_row - 10), (pc_col - 10, pc_row + 10), (0, 0, 255), 1)
cv2.line(marker_im, (pc_col - 10, pc_row + 10), (pc_col + 10, pc_row + 10), (0, 0, 255), 1)

area = ((pc_col+10)-(pc_col-10)) * ((pc_row+10)-(pc_row-10))
print('Centre ROI Area =', area)
area_aim = 20*20
if area != area_aim:
    print('Signal ROI area is too large/too small')
    sys.exit()

# draw bounding box around phantom
bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
if show_bbox:
    cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], bbox[2]), (255, 255, 255), 1)
    cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[3], bbox[2]), (255, 255, 255), 1)
    cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], bbox[0]), (255, 255, 255), 1)
    cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[1], bbox[0]), (255, 255, 255), 1)

""" DEFINE 4 QUADRANTS (COL,ROW) """
if show_quad:
    # top left
    cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], pc_row), (0, 0, 255), 1)
    cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col, bbox[0]), (bbox[1], bbox[0]), (0, 0, 255), 1)
    # top right
    cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[3], pc_row), (100, 200, 0), 1)
    cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (100, 200, 0), 1)
    cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (100, 200, 0), 1)
    cv2.line(marker_im, (pc_col, bbox[0]), (bbox[3], bbox[0]), (100, 200, 0), 1)
    # bottom left
    cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[1], pc_row), (0, 140, 255), 1)
    cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 140, 255), 1)
    cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (0, 140, 255), 1)
    cv2.line(marker_im, (pc_col, bbox[2]), (bbox[1], bbox[2]), (0, 140, 255), 1)
    # bottom right
    cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], pc_row), (255, 0, 0), 1)
    cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (255, 0, 0), 1)
    cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (255, 0, 0), 1)
    cv2.line(marker_im, (pc_col, bbox[2]), (bbox[3], bbox[2]), (255, 0, 0), 1)

""" PUT OTHER 4 ROIs IN CENTRE OF EACH QUADRANT """  # bbox (0min_row, 1min_col, 2max_row, 3max_col)
# centre coords for each quadrant
centre1 = [int(((pc_row - bbox[0])/2) + bbox[0]), int(((pc_col - bbox[1])/2) + bbox[1])]
centre2 = [int(((pc_row - bbox[0])/2) + bbox[0]), int(((bbox[3] - pc_col)/2) + pc_col)]
centre3 = [int(((bbox[2] - pc_row)/2) + pc_row), int(((pc_col - bbox[1])/2) + bbox[1])]
centre4 = [int(((bbox[2] - pc_row)/2) + pc_row), int(((pc_col - bbox[1])/2) + pc_col)]

# top left
cv2.line(marker_im, (centre1[1] + 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] - 10), (0, 0, 255), 1)
cv2.line(marker_im, (centre1[1] + 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] - 10), (0, 0, 255), 1)
cv2.line(marker_im, (centre1[1] - 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] + 10), (0, 0, 255), 1)
cv2.line(marker_im, (centre1[1] - 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] + 10), (0, 0, 255), 1)
# top right
cv2.line(marker_im, (centre2[1] + 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] - 10), (100, 200, 0), 1)
cv2.line(marker_im, (centre2[1] + 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] - 10), (100, 200, 0), 1)
cv2.line(marker_im, (centre2[1] - 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] + 10), (100, 200, 0), 1)
cv2.line(marker_im, (centre2[1] - 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] + 10), (100, 200, 0), 1)
# bottom left
cv2.line(marker_im, (centre3[1] + 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] - 10), (0, 140, 255), 1)
cv2.line(marker_im, (centre3[1] + 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] - 10), (0, 140, 255), 1)
cv2.line(marker_im, (centre3[1] - 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] + 10), (0, 140, 255), 1)
cv2.line(marker_im, (centre3[1] - 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] + 10), (0, 140, 255), 1)
# bottom right
cv2.line(marker_im, (centre4[1] + 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] - 10), (255, 0, 0), 1)
cv2.line(marker_im, (centre4[1] + 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] - 10), (255, 0, 0), 1)
cv2.line(marker_im, (centre4[1] - 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] + 10), (255, 0, 0), 1)
cv2.line(marker_im, (centre4[1] - 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] + 10), (255, 0, 0), 1)

if show_graphical:
    cv2.imshow('Signal ROIs', marker_im)
    cv2.waitKey(0)

# signal values corresponding to voxels inside each signal ROI
signal0 = np.mean(img[pc_row-10:pc_row+10, pc_col-10:pc_col+10])
signal1 = np.mean(img[centre1[0]-10:centre1[0]+10, centre1[1]-10:centre1[1]+10])
signal2 = np.mean(img[centre2[0]-10:centre2[0]+10, centre2[1]-10:centre2[1]+10])
signal3 = np.mean(img[centre3[0]-10:centre3[0]+10, centre3[1]-10:centre3[1]+10])
signal4 = np.mean(img[centre4[0]-10:centre4[0]+10, centre4[1]-10:centre4[1]+10])

print('Mean signal in each of the 5 ROIs =', signal0, signal1, signal2, signal3, signal4)

# draw background ROIs

"""FROM AMENDED FROM BASIC IMAGE TEST"""
# SNR measure
factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
mean_phantom = np.mean([signal0, signal1, signal2, signal3, signal4])  # mean signal from image data (not filtered!)
print('Mean signal (total) =', mean_phantom)

# auto detection of 4 x background ROI samples (one in each corner of background)
bound_box_mask = np.zeros(np.shape(img))
bound_box_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255

anti_bound_box_mask = 255-bound_box_mask
background_mask = 255-mask

if show_graphical:
    cv2.imshow('Phantom Bounding Box Mask', bound_box_mask)
    cv2.waitKey(0)
    cv2.imshow('Anti BBOX Mask', anti_bound_box_mask)
    cv2.waitKey(0)
    cv2.imshow('Phantom Mask', mask)
    cv2.waitKey(0)
    cv2.imshow('Background Mask', background_mask)
    cv2.waitKey(0)

idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
rows = idx[0]
cols = idx[1]
min_row = np.min(rows)  # first row of phantom
max_row = np.max(rows)  # last row of phantom

min_col = np.min(cols)  # first column of phantom
max_col = np.max(cols)  # last column of phantom

half_row = int(dims[0] / 2)  # half way row
mid_row1 = int(round(min_row / 2))
mid_row2 = int(round((((dims[0] - max_row) / 2) + max_row)))

half_col = int(dims[1] / 2)  # half-way column
mid_col1 = int(round(min_col / 2))
mid_col2 = int(round((((dims[1] - max_col) / 2) + max_col)))

bROI1 = np.zeros(np.shape(mask))  # initialise image matrix for each corner ROI
bROI2 = np.zeros(np.shape(mask))
bROI3 = np.zeros(np.shape(mask))
bROI4 = np.zeros(np.shape(mask))
bROI5 = np.zeros(np.shape(mask))

# Background ROIs according to MagNET protocol
# TODO: write funtion that checks every ROI is 20x20, not covering phantom and fits within FOV
if caseT:
    bROI1[mid_row1-10:mid_row1+10, min_col-10:min_col+10] = 255  # top left
    marker_im[mid_row1-10:mid_row1+10, min_col-10:min_col+10] = (0, 0, 255)
    bROI1_check = sf.check_ROI(bROI1, bin_mask)

    bROI2[mid_row1-10:mid_row1+10, max_col-10:max_col+10] = 255  # top right
    marker_im[mid_row1-10:mid_row1+10, max_col-10:max_col+10] = (0, 255, 0)
    bROI2_check = sf.check_ROI(bROI2, bin_mask)

    bROI3[mid_row2-30:mid_row2-10, min_col-10:min_col+10] = 255  # bottom left
    marker_im[mid_row2-30:mid_row2-10, min_col-10:min_col+10] = (255, 0, 0)
    bROI3_check = sf.check_ROI(bROI3, bin_mask)

    # TODO: check that this fits below phantom.... if not then place on top of phantom
    bROI4[mid_row2-10:mid_row2+10, pc_col-10:pc_col+10] = 255  # bottom centre
    marker_im[mid_row2-10:mid_row2+10, pc_col-10:pc_col+10] = (0, 140, 255)
    bROI4_check = sf.check_ROI(bROI4, bin_mask)

    bROI5[mid_row2-30:mid_row2-10, max_col-10:max_col+10] = 255  # bottom right
    marker_im[mid_row2-30:mid_row2-10, max_col-10:max_col+10] = (205, 235, 255)
    bROI5_check = sf.check_ROI(bROI5, bin_mask)

if caseS or caseC:
    bROI1[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col-5] = 255  # top left
    marker_im[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col-5] = (0, 0, 255)
    bROI1_check = sf.check_ROI(bROI1, bin_mask)

    bROI2[mid_row1-10:mid_row1+10, max_col+5:max_col+25] = 255  # top right
    marker_im[mid_row1-10:mid_row1+10, max_col+5:max_col+25] = (0, 255, 0)
    bROI2_check = sf.check_ROI(bROI2, bin_mask)

    bROI3[mid_row2-10:mid_row2+10, min_col-25:min_col-5] = 255  # bottom left
    marker_im[mid_row2-10:mid_row2+10, min_col-25:min_col-5] = (255, 0, 0)
    bROI3_check = sf.check_ROI(bROI3, bin_mask)

    # TODO: check that this fits below phantom.... if not then place on top of phantom
    bROI4[mid_row2-10:mid_row2+10, pc_col-10:pc_col+10] = 255  # bottom centre
    marker_im[mid_row2-10:mid_row2+10, pc_col-10:pc_col+10] = (0, 140, 255)
    bROI4_check = sf.check_ROI(bROI4, bin_mask)

    bROI5[mid_row2-10:mid_row2+10, max_col+5:max_col+25] = 255  # bottom right
    marker_im[mid_row2-10:mid_row2+10, max_col+5:max_col+25] = (205, 235, 255)
    bROI5_check = sf.check_ROI(bROI5, bin_mask)

if show_graphical:
    cv2.imshow('Signal and Background ROIs', marker_im)
    cv2.waitKey(0)

# stop here for now
sys.exit()

# background/noise voxel values
n1 = np.std(img[mid_row1-10:mid_row1+10, min_col-10:min_col+10])
n2 = np.std(img[mid_row1-10:mid_row1+10, max_col-10:max_col+10])
n3 = np.std(img[mid_row2-30:mid_row2-10, min_col-10:min_col+10])
n4 = np.std(img[mid_row2-10:mid_row2+10, pc_col-10:pc_col+10])
n5 = np.std(img[mid_row2-30:mid_row2-10, max_col-10:max_col+10])

print('Noise (standard deviation of signal in each background ROI) = ', n1, n2, n3, n4, n5)

noise = np.mean([n1, n2, n3, n4, n5])

print('Noise (total) = ', noise)

# SNR calculation (background method as opposed to subtraction method)
SNR_background = (factor * mean_phantom) / noise
print('SNR = ', SNR_background.round(2))

# TODO: normalised SNR calculation - need to extract DICOM header data
# END












