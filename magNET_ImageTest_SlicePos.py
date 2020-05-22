"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

from DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
import imutils

from scipy.spatial import distance as dist
from imutils import perspective
import argparse
from skimage.measure import profile_line, label, regionprops

from nibabel.viewers import OrthoSlicer3D # << actually do use this!!
from imutils import contours
from skimage import filters, segmentation
from skimage.morphology import binary_erosion, convex_hull_image
from skimage import exposure as ex


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


directpath = "MagNET_acceptance_test_data/scans/"
folder = "42-SLICE_POS"

pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

with os.scandir(pathtodicom) as it:
    for file in it:
        path = "{0}{1}".format(pathtodicom, file.name)

ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py


def slice_pos_meta(dicomfile):
    # extract metadata for slice postion info calculations
    # dicomfile = pydicom.dataset.FileDataset
    elem = dicomfile[0x5200, 0x9230]  # pydicom.dataelem.DataElement, (Per-frame Functional Groups Sequence)
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        st = xx.SliceThickness
        slice_space = xx.SpacingBetweenSlices
        pixels_space = xx.PixelSpacing

    return st, slice_space, pixels_space


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

    conv_hull = convex_hull_image(ots)  # set of pixels included in smallest convex polygon that SURROUND all white pixels in the input image
    ch = np.multiply(conv_hull, 1)  # bool --> binary

    fore_image = ch * img  # phantom
    back_image = (1 - ch) * img  #background

    if slice_dim == 0:
        mask3D[imslice, :, :] = ch
    if slice_dim == 1:
        mask3D[:, imslice, :] = ch
    if slice_dim == 2:
        mask3D[:, :, imslice] = ch

#OrthoSlicer3D(mask3D).show()  # look at 3D volume data

# For slice position analysis want to do analysis on every slice but for now start with mid-slice
# TODO: make this code work for every slice! and make measurement
# for zz in range(no_slices):
zz = int(round(no_slices/2))  # slice of interest
phmask = mask3D[zz, :, :]  # phantom mask
phim = imdata[zz, :, :]*phmask  # phantom image
bgim = imdata[zz, :, :]*~phmask  # background image

# display image
cv2.imshow('phantom image', ((phim/np.max(phim))*255).astype('uint8'))
cv2.waitKey(0)

phim_dims = np.shape(phim)

phim_norm = phim/np.max(phim)
phim_gray = phim_norm*255

edged = cv2.Canny(phim_gray.astype('uint8'), 20, 200)
bigbg = cv2.dilate(~phmask.astype('uint8'), None, iterations=4)  # dilate background mask
bigbg[bigbg == 254] = 0

edged = edged*~bigbg

cv2.imshow('Dilated Background', bigbg)
cv2.waitKey(0)

cv2.imshow('Canny Filter', edged.astype('float32'))
cv2.waitKey(0)

edgedd = cv2.dilate(edged, None, iterations=1)

cv2.imshow('Canny Dilated', edgedd.astype('float32'))
cv2.waitKey(0)

edgede = cv2.erode(edgedd, None, iterations=1)

cv2.imshow('Canny Eroded', edgede.astype('float32'))
cv2.waitKey(0)

lines_im = phmask.copy()

minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edgede, 1, np.pi/180, 5, minLineLength, maxLineGap)

no_lines = lines.shape
no_lines = no_lines[0]
print('The number of lines detected is = ', no_lines)

for lineno in np.linspace(0, no_lines-1, no_lines, dtype=int):
    for x1, y1, x2, y2 in lines[lineno]:
        cv2.line(lines_im, (x1, y1), (x2, y2), 0, 2)

label_this = edgede*lines_im

cv2.imshow('Rods Detected', label_this.astype('float32'))
cv2.waitKey(0)

label_img, num = label(label_this, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
print('Number of regions detected (should be 6!!!) = ', num)

cv2.imshow('Rods labelled', label_img.astype('float32'))
cv2.waitKey(0)

props = regionprops(label_img)  # returns region properties for labelled image
cent = np.zeros([num, 2])

marker_im = phmask.copy()

for xx in range(num):
    cent[xx, :] = props[xx].centroid  # central coordinate

cent = np.round(cent).astype(int)

for i in cent:
    # draw the center of the circle
    cv2.circle(marker_im, (i[0], i[1]), 1, 0, 1)

marker_im = marker_im*255
cv2.imshow('marker image', marker_im.astype('uint8'))
cv2.waitKey(0)

temp1 = []
temp2 = []
temp3 = []

for i in cent:
    if i[1] < 70:
        temp1.append(i)
    if 100 < i[1] < 150:
        temp2.append(i)
    if i[1] > 180:
        temp3.append(i)

# top 2 parallel rods
src1 = temp1[0]
src1 = (src1[0], src1[1])
dst1 = temp1[1]
dst1 = (dst1[0], dst1[1])

# angled rods
src2 = temp2[0]
src2 = (src2[0], src2[1])
dst2 = temp2[1]
dst2 = (dst2[0], dst2[1])

# bottom 2 parallel rods
src3 = temp3[0]
src3 = (src3[0], src3[1])
dst3 = temp3[1]
dst3 = (dst3[0], dst3[1])


#TODO: repalce marker_im with RGB/grayscale image of phantom and draw coloured lines on image instead
hmarker_im = marker_im.copy()  # for horizontal lines
vmarker_im = marker_im.copy()  # for vertical lines

cv2.line(hmarker_im, src1, dst1, 0, 1)
# cv2.line(hmarker_im, src2, dst2, 0, 1)  # diagonal line between angled rods
cv2.line(hmarker_im, (src2[0], int(phim_dims[1]/2)), (dst2[0], int(phim_dims[1]/2)), 0, 1)  # horizontal lines between angled rods
cv2.line(hmarker_im, src3, dst3, 0, 1)

cv2.imshow('horiz. marker image', hmarker_im.astype('uint8'))
cv2.waitKey(0)

cv2.line(vmarker_im, src1, src3, 0, 1)
cv2.line(vmarker_im, dst1, dst3, 0, 1)
#TODO: replace centre of FOV with centre of full phantom
cv2.line(vmarker_im, (int(phim_dims[0]/2), src2[1]), (int(phim_dims[0]/2), dst2[1]), 0, 1)

cv2.imshow('vert. marker image', vmarker_im.astype('uint8'))
cv2.waitKey(0)

#compute the Euclidean distance between the midpoints
# horizontal lines
hdist1 = dist.euclidean(src1, dst1)
hdist2 = dist.euclidean((src2[0], int(phim_dims[1]/2)), (dst2[0], int(phim_dims[1]/2)))
hdist3 = dist.euclidean(src3, dst3)

print('Horizontal distance (top) = ', hdist1, 'mm')
print('Horizontal distance between angled rods = ', hdist2, 'mm')
print('Horizontal distance (bottom) = ', hdist3, 'mm')

# horizontal lines
cv2.putText(hmarker_im, "{:.1f}mm".format(hdist1), (int(phim_dims[0]/2), int(dst1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
cv2.putText(hmarker_im, "{:.1f}mm".format(hdist2), (src2[0] + 30, int(phim_dims[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
cv2.putText(hmarker_im, "{:.1f}mm".format(hdist3), (int(phim_dims[0]/2), int(dst3[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

cv2.imshow("horiz. measurements", hmarker_im.astype('uint8'))
cv2.waitKey(0)

# vertical lines
vdist1 = dist.euclidean(src1, src3)
vdist2 = dist.euclidean(dst1, dst3)
#TODO: replace centre of FOV with centre of full phantom
vdist3 = dist.euclidean((int(phim_dims[0]/2), src2[1]), (int(phim_dims[0]/2), dst2[1]))

print('Vertical distance (left) = ', vdist1, 'mm')
print('Vertical distance (right) = ', vdist2, 'mm')
print('Vertical distance between angled rods = ', vdist3, 'mm')

cv2.putText(vmarker_im, "{:.1f}mm".format(vdist1), (int(src1[0] + 5), int(src3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
cv2.putText(vmarker_im, "{:.1f}mm".format(vdist2), (int(dst1[0] + 15), int(dst3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
cv2.putText(vmarker_im, "{:.1f}mm".format(vdist3), (int(phim_dims[0]/2), int(phim_dims[1]/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)

cv2.imshow("vert. measurements", vmarker_im.astype('uint8'))
cv2.waitKey(0)

# CALCULATIONS
RDA = [120, 120, 120, 120]
RDM = [hdist1, hdist3, vdist1, vdist2]
#print(RDM)
CFall = np.divide(RDA, RDM)
CF = np.mean(CFall)
print('Mean Correction Factor = ', CF)

distance_of_angled_rods = zz#vdist3  # not sure is this the right value here according to Excel sheet.....
slice_position_measured = np.sqrt(((distance_of_angled_rods*CF)**2) - (6.5**2))/2
print('Measured position = ', slice_position_measured)

first_slice_position = 1  # get this from header info? is this slice number...?
slice_position = zz+1
slice_thickness, slice_gap, pixeldims = slice_pos_meta(ds)

print(slice_thickness, slice_gap, pixeldims)

# this changes with respect to previous position (see Excel sheet!) - will make more sense will extend code to 3D loop
# over slices
slice_position_calculated_upper = first_slice_position + ((slice_position - 1)*slice_thickness)
slice_position_calculated_lower = first_slice_position - ((slice_position - 1)*slice_thickness)
slice_position_calculated = slice_position_calculated_upper - slice_position_calculated_lower  # this is not correct.
print('Calculated position = ', slice_position_calculated)

slice_position_error = slice_position_calculated - slice_position_measured
print('Error = ', slice_position_error, 'mm (must be within +- 1 mm)')

