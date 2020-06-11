"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import resolution_funcs as rf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
import re
import scipy
from scipy import fft, ifft, interpolate, ndimage

from DICOM_test import dicom_read_and_write
from skimage import filters
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!
from skimage.morphology import opening

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/Resolution_Images/"

show_graphical = True

with os.scandir(directpath) as the_folders:
    for folder in the_folders:
        fname = folder.name
        if re.search('-RES_', fname):
            print(folder.name)
            # TODO: iterate between [256, 512] versions of ['COR', 'SAG', 'TRA'] and repeat analysis
            # FIRST ATTEMPT: COR 256 ANALYSIS
            if re.search('512', fname):
                if re.search('_COR_', fname):
                    folder = fname
                    pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                    with os.scandir(pathtodicom) as it:
                        for file in it:
                            path = "{0}{1}".format(pathtodicom, file.name)

                    ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

try:
    xdim, ydim = dims
    print('Matrix Size =', xdim, 'x', ydim)

    img = ((imdata/np.max(imdata))*255).astype('uint8')  # grayscale

    cv2.imshow('dicom imdata', img)
    cv2.waitKey(0)

except ValueError:
    print('DATA INPUT ERROR: this is 3D image data')
    OrthoSlicer3D(imdata).show()  # look at 3D volume data
    sys.exit()


# create mask
mask = rf.create_2D_mask(img)  # watch out for grayscale mask!! [0, 255]
bin_mask = (mask/np.max(mask)).astype('uint8')  # binary mask [0, 1]

if show_graphical:
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

phim = img*bin_mask  # phantom masked image
bgim = img*(1-bin_mask)  # background image

# cv2.imshow('phantom masked', phim)
# cv2.waitKey(0)
# cv2.imshow('background masked', bgim)
# cv2.waitKey(0)

# otsu threshold
ots = np.zeros_like(phim, dtype=np.uint8)
ots[(phim > filters.threshold_otsu(phim)) == True] = 255  # Otsu threshold on weighted combination

if show_graphical:
    cv2.imshow('Otsu Threshold', ots)
    cv2.waitKey(0)

erode_mask = cv2.erode(bin_mask, None, iterations=1)

sections_only = (255-ots)*erode_mask

dilate_sections = cv2.dilate(sections_only, None, iterations=2)
erode_sections = cv2.erode(dilate_sections, None, iterations=2)
label_this = erode_sections

if show_graphical:
    cv2.imshow('Label This', label_this)
    cv2.waitKey(0)

# label this
label_img, num = label(label_this, connectivity=ots.ndim, return_num=True)  # labels the mask
print('Number of regions detected = ', num)

if show_graphical:
    plt.figure()
    plt.imshow(label_img)
    plt.show()

props = regionprops(label_img)  # returns region properties for labelled image
cent = np.zeros([num, 2])
areas = np.zeros([num, 1])
mxAL = np.zeros([num, 1])
mnAL = np.zeros([num, 1])

# show detected regions and lines on marker_im
marker_im = phim.copy()
marker_im = marker_im.astype('uint8')
marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

for xx in range(num):
    cent[xx, :] = props[xx].centroid  # central coordinate
    areas[xx, :] = props[xx].area  # area of detected region
    mxAL[xx, :] = props[xx].major_axis_length
    mnAL[xx, :] = props[xx].minor_axis_length

cent = cent.astype(int)
idx = np.where(areas == np.max(areas))
idx = int(idx[0])
print('Label number to discard = ', idx+1)
label_idxs_all = np.arange(1, num+1)  # list of label numbers - including central section
label_idxs_where = np.where(label_idxs_all != idx+1)  # labels to be kept
label_idxs = label_idxs_all[label_idxs_where]  # identify labels to be kept

sb = 0  # for subplotting!

for ii in label_idxs:
    print(ii)
    block_im = np.zeros(np.shape(label_img))
    block_im[label_img == ii] = 255

    if show_graphical:
        cv2.imshow('Central Block', block_im.astype('uint8'))
        cv2.waitKey(0)
        print('Major axis length = ', mxAL[ii-1, :])
        print('Minor axis length = ', mnAL[ii-1, :])

    elements = np.where(block_im == 255)  # [y, x], [rows, cols]

    # top right corner
    min_row = np.min(elements[0])

    # bottom corner
    max_row = np.max(elements[0])

    # top left corner
    min_col = np.min(elements[1])

    # right corner
    max_col = np.max(elements[1])

    # cross section lines
    half_col = int(min_col + ((max_col - min_col)/2))
    half_row = int(min_row + ((max_row - min_row)/2))

    # draw the bounding box
    cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
    cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
    cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
    cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

    #major and minor axes
    dist1 = dist.euclidean((half_col, max_row), (half_col, min_row))  # vertical distance
    dist2 = dist.euclidean((min_col, half_row), (max_col, half_row))  # horizontal distance

    if dist1 < dist2:  # vertical line is minor axis
        case1 = True
        case2 = False
        print('Phase encoding direction.')  # TODO: check this is the case?
    if dist1 > dist2:  # horizontal line is minor axis
        case1 = False
        case2 = True
        print('Frequency encoding direction.')
    else:
        case1 = False
        case2 = False
        ValueError

    xtrabit = 10  # amount the line is extended over edge to get sense of baseline values

    if case1:
        cv2.line(marker_im, (half_col, max_row+xtrabit), (half_col, min_row-xtrabit), (255, 0, 0), 1)  # draw vertical line
        src = (half_col, max_row+xtrabit)
        dst = (half_col, min_row-xtrabit)
    if case2:
        src = (min_col-xtrabit, half_row)
        dst = (max_col+xtrabit, half_row)
        cv2.line(marker_im, (min_col-xtrabit, half_row), (max_col+xtrabit, half_row), (255, 0, 0), 1)  # horizontal line

    if show_graphical:
        #cv2.imwrite("{0}marker_image.png".format(imagepath), marker_im.astype('uint8'))
        cv2.imshow('marker image', marker_im.astype('uint8'))
        cv2.waitKey(0)

    """DRAW LINE PROFILE ACROSS PARALLEL BARS"""
    print('Source = ', src)
    print('Destination = ', dst)

    linewidth = 1  # width of the line (mean taken over width)

    # display profile line on phantom: from source code of profile_line function
    src_col, src_row = src = np.asarray(src, dtype=float)
    dst_col, dst_row = dst = np.asarray(dst, dtype=float)
    d_col, d_row = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # add one above to include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row])
    perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col])

    improfile = np.copy(img)
    improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 255

    # plot sampled line on phantom to visualise where output comes from
    # plt.figure()
    # plt.imshow(improfile)
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()

    # voxel values along specified line. specify row then column (y then x)
    output = profile_line(img, (src[1], src[0]), (dst[1], dst[0]))  # signal profile

    baseline_vals = np.append(output[0:xtrabit], output[-(xtrabit+1):])
    base_signal = np.repeat(np.mean(baseline_vals), len(output))
    min_signal = np.repeat(np.min(output), len(output))
    signal50 = min_signal + ((base_signal-min_signal)/2)

    sb = sb + 1

    plt.figure(1)
    plt.subplot(2, 2, sb)
    plt.plot(output)
    plt.plot(base_signal, 'g--')
    plt.plot(signal50, 'k--')
    plt.plot(min_signal, 'r--')
    plt.legend(['Contrast Response Function (CRF)', 'Baseline Signal', '50% Signal', 'Minimum Signal'],
               fontsize='xx-small', loc='lower left')

plt.show()

# TODO: confirm pixel dimensions as well as matrix size from dicom header
# TODO: identify 4 black line values, determine if abover or below 50% threshold.
# TODO: evaluate if pass/fail
# TODO: apply to other geometries

sys.exit()

