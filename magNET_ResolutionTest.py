"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import resolution_funcs as rf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
import re

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
print(mask.dtype, np.min(mask), np.max(mask))
bin_mask = (mask/np.max(mask)).astype('uint8')  # binary mask
print(bin_mask.dtype, np.min(bin_mask), np.max(bin_mask))

cv2.imshow('mask', mask)
cv2.waitKey(0)

phim = img*bin_mask  # phantom masked image
bgim = img*(1-bin_mask)  # background image

cv2.imshow('phantom masked', phim)
cv2.waitKey(0)
cv2.imshow('background masked', bgim)
cv2.waitKey(0)

ots = np.zeros_like(phim, dtype=np.uint8)
ots[(phim > filters.threshold_otsu(phim)) == True] = 255  # Otsu threshold on weighted combination

cv2.imshow('Otsu Threshold', ots)
cv2.waitKey(0)

erode_mask = cv2.erode(bin_mask, None, iterations=1)

label_this = (255-ots)*erode_mask

cv2.imshow('Label This', label_this)
cv2.waitKey(0)

label_img, num = label(label_this, connectivity=ots.ndim, return_num=True)  # labels the mask

# plt.figure()
# plt.imshow(label_img)
# plt.show()

print('Number of regions detected = ', num)

props = regionprops(label_img)  # returns region properties for labelled image
cent = np.zeros([num, 2])
areas = np.zeros([num, 1])

marker_im = phim.copy()
marker_im = marker_im.astype('uint8')
marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

for xx in range(num):
    cent[xx, :] = props[xx].centroid  # central coordinate
    areas[xx, :] = props[xx].area  # area of detected region

cent = cent.astype(int)
idx = np.where(areas == np.max(areas))
idx = int(idx[0])

bboxx = props[idx].bbox  # bounding box coordinates

min_row, min_col, max_row, max_col = bboxx
half_col = int(min_col + ((max_col-min_col)/2))
half_row = int(min_row + ((max_row-min_row)/2))
# draw the bounding box
cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

cv2.line(marker_im, (half_col, max_row), (half_col, min_row), (255, 0, 0), 1)
cv2.line(marker_im, (min_col, half_row), (max_col, half_row), (255, 0, 0), 1)

if show_graphical:
    cv2.imwrite("{0}marker_image.png".format(imagepath), marker_im.astype('uint8'))
    cv2.imshow('marker image', marker_im.astype('uint8'))
    cv2.waitKey(0)

# TODO: Add Hough Lines Detection and measure angle between lines.

sys.exit()



#         label_this2 = label_this.copy()
#         minLineLength = 2
#         maxLineGap = 6
#         theta = np.pi  # 0 degrees to detect vertical lines
#         label_this2 = label_this2.astype('uint8')
#         lines_im2 = phmask.copy()
#         lines = cv2.HoughLinesP(label_this2, 1, theta, 5, minLineLength, maxLineGap)
#
#         no_lines = lines.shape
#         no_lines = no_lines[0]
#
#         for lineno in np.linspace(0, no_lines - 1, no_lines, dtype=int):
#             for x1, y1, x2, y2 in lines[lineno]:
#                 cv2.line(lines_im2, (x1, y1), (x2, y2), 0, 2)
#
#         label_this3 = label_this2 * lines_im2
#
#         if show_graphical:
#             cv2.imwrite("{0}lineremoval2_slice_{1}.png".format(imagepath, zz + 1), (label_this3*255).astype('uint8'))
#             cv2.imshow('After 2nd line removal', label_this3.astype('float32'))
#             cv2.waitKey(0)
#
#         label_img2, num2 = label(label_this3, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
#         print('Number of regions detected (should be 6!!!) = ', num2)
#
#         if num2 > 6:
#             print('Still too many regions detected! =(')
#
#         label_this = label_this3
#         num = num2
#         label_img = label_img2  # replace label_img with new version with less labelled regions
#
#     if show_graphical:
#         cv2.imwrite("{0}labelled_rods_slice_{1}.png".format(imagepath, zz + 1), (label_img*255).astype('uint8'))
#         cv2.imshow('Rods labelled', label_img.astype('float32'))
#         cv2.waitKey(0)
#
#     props = regionprops(label_img)  # returns region properties for labelled image





