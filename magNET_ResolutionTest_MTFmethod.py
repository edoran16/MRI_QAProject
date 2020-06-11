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

cv2.imshow('mask', mask)
cv2.waitKey(0)

phim = img*bin_mask  # phantom masked image
bgim = img*(1-bin_mask)  # background image

cv2.imshow('phantom masked', phim)
cv2.waitKey(0)
cv2.imshow('background masked', bgim)
cv2.waitKey(0)

# otsu threshold
ots = np.zeros_like(phim, dtype=np.uint8)
ots[(phim > filters.threshold_otsu(phim)) == True] = 255  # Otsu threshold on weighted combination

cv2.imshow('Otsu Threshold', ots)
cv2.waitKey(0)

erode_mask = cv2.erode(bin_mask, None, iterations=1)

label_this = (255-ots)*erode_mask

cv2.imshow('Label This', label_this)
cv2.waitKey(0)

# label this
label_img, num = label(label_this, connectivity=ots.ndim, return_num=True)  # labels the mask
print('Number of regions detected = ', num)
plt.figure()
plt.imshow(label_img)
plt.show()

sys.exit()

props = regionprops(label_img)  # returns region properties for labelled image
cent = np.zeros([num, 2])
areas = np.zeros([num, 1])

# show detected regions and lines on marker_im
marker_im = phim.copy()
marker_im = marker_im.astype('uint8')
marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

for xx in range(num):
    cent[xx, :] = props[xx].centroid  # central coordinate
    areas[xx, :] = props[xx].area  # area of detected region

cent = cent.astype(int)
idx = np.where(areas == np.max(areas))
idx = int(idx[0])
box_region = props[idx]
print('Label number to segment = ', idx+1)

central_block_im = np.zeros(np.shape(label_img))
central_block_im[label_img == 4] = 255

cv2.imshow('Central Block', central_block_im.astype('uint8'))
cv2.waitKey(0)

# dilate the central block so that the signal profile can extend before and after the detected edge
# dilated_block = cv2.dilate(central_block_im, None, iterations=5)
# central_block_im = dilated_block
# TODO: think of way to extend beyond the perimeter at angle
# cv2.imshow('Dilated Central Block', central_block_im.astype('uint8'))
# cv2.waitKey(0)

elements = np.where(central_block_im == 255)  # [y, x], [rows, cols]

# top corner
min_row = np.min(elements[0])
min_row_col_idx = np.where(elements[0] == min_row)
min_row_col = np.max(elements[1][min_row_col_idx])  # right most column!!!

# bottom corner
max_row = np.max(elements[0])
max_row_col_idx = np.where(elements[0] == max_row)
max_row_col = np.min(elements[1][max_row_col_idx])  # right most column!!!

# left corner
min_col = np.min(elements[1])
min_col_row_idx = np.where(elements[1] == np.min(elements[1]))
min_col_row = np.min(elements[0][min_col_row_idx])

# right corner
max_col = np.max(elements[1])
max_col_row_idx = np.where(elements[1] == np.max(elements[1]))
max_col_row = np.max(elements[0][max_col_row_idx])

# horizontal line at angle
half_col_left = int(min_col + ((max_row_col - min_col)/2))
half_row_left = int(min_col_row + ((max_row - min_col_row)/2))

half_col_right = int(min_row_col + ((max_col - min_row_col)/2))
half_row_right = int(min_row + ((max_col_row - min_row)/2))

# vertical line at angle
half_col_top = int(min_col + ((min_row_col - min_col)/2))
half_row_top = int(min_row + ((min_col_row - min_row)/2))

half_col_bottom = int(max_row_col + ((max_col - max_row_col)/2))
half_row_bottom = int(max_col_row + ((max_row - max_col_row)/2))

# draw the bounding box
cv2.line(marker_im, (min_row_col, min_row), (max_col, max_col_row), (0, 0, 255), 1)
cv2.line(marker_im, (max_col, max_col_row), (max_row_col, max_row), (0, 0, 255), 1)
cv2.line(marker_im, (max_row_col, max_row), (min_col, min_col_row), (0, 0, 255), 1)
cv2.line(marker_im, (min_col, min_col_row), (min_row_col, min_row), (0, 0, 255), 1)

cv2.line(marker_im, (half_col_left, half_row_left), (half_col_right, half_row_right), (255, 0, 0), 1)
cv2.line(marker_im, (half_col_top, half_row_top), (half_col_bottom, half_row_bottom), (255, 0, 0), 1)

# """ Rotate image"""
# cv2.imshow('img', img.astype('uint8'))
# cv2.waitKey(0)
# # rotation angle in degree
# rotated = ndimage.rotate(img, 350)
# cv2.imshow('rotated img', rotated.astype('uint8'))
# cv2.waitKey(0)

if show_graphical:
    cv2.imwrite("{0}marker_image.png".format(imagepath), marker_im.astype('uint8'))
    cv2.imshow('marker image', marker_im.astype('uint8'))
    cv2.waitKey(0)

"""DRAW LINE PROFILE FOR EDGE RESPONSE FUNCTION"""
# TODO: check if line has to perpendicular to edge, or in FE/PE direction
# VERTICAL LINE ANGLED
# src = (half_col, min_row)  # starting point x, y coordinate (column, row)
# dst = (half_col, max_row)  # finish point x, y coordinate (column, row)

# HORIZONTAL LINE ANGLED
m = (half_row_right - half_row_left)/(half_col_right - half_col_left)
print('Gradient = ', m)
c = half_row_left - (m*half_col_left)  # y - mx
print('Intercept = ', c)

src_x = half_col_left-10
src_y = (m*src_x) + c
src = (src_x, src_y)
# src = (half_col_left, half_row_left)  # starting point x, y coordinate (column, row)

dst_x = half_col_left+10
dst_y = (m*dst_x) + c
dst = (dst_x, dst_y)
# dst = (half_col_right, half_row_right)  # finish point x, y coordinate (column, row)
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

"""EDGE RESPONSE FUNCTION"""
# voxel values along specified line. specify row then column (y then x)
output = profile_line(img, (src[1], src[0]), (dst[1], dst[0]))  # signal profile
voxels = np.arange(0, len(output)) - int(len(output)/2)   # define 0 mm as edge cross section
soutput = rf.smooth(output, window_len=11)  # smoothed signal profile

# plt.figure()
# plt.plot(voxels, output, voxels, soutput)
# plt.legend(['Signal Profile', 'Smoothed Signal Profile'])
# plt.show()

# FOLLOWING THE ANALYSIS IN THIS PAPER vv
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6236841/pdf/ACM2-19-244.pdf
f = interpolate.interp1d(voxels, soutput)  # spline interpolation
voxnew = np.linspace(0, len(soutput)-1, 1000) - int(len(soutput)/2)  # define new voxel/distance vector
outnew = f(voxnew)   # use interpolation function returned by `interp1d`

# plt.figure()
# plt.plot(voxels, soutput, 'o', voxnew, outnew, '.')
# plt.legend(['Smoothed Signal Profile', 'Spline Interpolation'])
# plt.show()

outnorm = outnew/np.max(outnew)

""" Least squares fit of ERF """
from scipy.optimize import least_squares, curve_fit

ys = outnorm
xs = voxnew


def fit_func(x, a, b):
    a_in_rad = np.deg2rad(a)  # a in radians
    tan_a_in_rad = np.tan(a_in_rad)  # tan(a)
    arctan_xa_product = b*np.arctan(-x * tan_a_in_rad)  # arctan(x*tan(a))
    arctan_xa_product = arctan_xa_product + np.abs(np.min(arctan_xa_product))  # zero function
    arctan_xa_product = arctan_xa_product/np.max(arctan_xa_product)  # normalise function
    return arctan_xa_product


a0 = 45
b0 = 0.5
param, param_cov = curve_fit(fit_func, xs, ys, p0=(a0, b0), bounds=([30, 0.4], [60, 0.6]))
print('Using curve we retrieve a value of a = ', param[0], 'b = ', param[1])

y = fit_func(xs, param[0], param[1])
plt.figure(1)
plt.plot(xs, ys)
plt.plot(xs, y)
plt.legend(['ERF Data', 'arctan Fit to ERF'])
plt.show()

# ERF = ys  # smoothed and interpolated data
ERF = y  # arctan data

# ERF
plt.figure(2)
plt.subplot(131)
plt.plot(voxnew, ERF)
plt.xlabel('Position (mm)')
plt.title('Edge Response Function')

"""DIFFERENTIATE FOR LINE SPREAD FUNCTION"""
# TODO: differentiate properly...
# from sympy import *
# x, y, z = symbols('x y z')
# #f = z*(1/(tan(-x * y)))  # arctan(x*tan(a))
# f = z*(1/tan(-x*y))
# df = diff(f, x)
#
# print(df)
#
# xx = voxnew
# yy = np.tan(np.deg2rad(param[0]))
# zz = param[1]
#
# diff_lsf = yy*zz*(np.tan(xx*yy)**2 + 1)/np.tan(xx*yy)**2
#
# plt.figure(3)
# plt.plot(voxnew, diff_lsf)

diff_output = []

for dd in np.linspace(0, len(ERF)-2, len(ERF)-1):
    diff = np.abs(ERF[int(dd+1)] - ERF[int(dd)])
    diff_output.append(diff)

diff_zeroed = diff_output - np.min(diff_output)  # zero the LSF
diff_norm = diff_zeroed/np.max(diff_zeroed)  # normalise LSF
LSF = diff_norm
voxs = np.linspace(np.min(voxnew), np.max(voxnew), len(diff_norm)) #- int(np.max(voxnew)/2)

# LSF
plt.figure(2)
plt.subplot(132)
plt.plot(voxs, LSF)
plt.xlabel('Position (mm)')
plt.title('Line Spread Function')

"""FOURIER TRANSFORM FOR MTF"""

MTF = np.fft.fftshift(fft(diff_norm))
# MTF = fft(LSF)  # fourier transform of LSF
MTF_abs = np.abs(MTF)  # absolute value norm of the FT
MTF_norm = MTF_abs/np.max(MTF_abs)  # normalised MTF

spatial_res = 1/voxs

# MTF
plt.figure(2)
plt.subplot(133)
#plt.plot(spatial_res, MTF_norm)
plt.plot(spatial_res, MTF_norm)
plt.xlabel('Spatial Frequency (cycles/mm)')
plt.title('Modulation Transfer Function')
plt.show()

sys.exit()




