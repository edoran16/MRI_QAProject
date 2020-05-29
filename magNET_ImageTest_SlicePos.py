"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import slice_pos_funcs as spf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2

from DICOM_test import dicom_read_and_write
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!
from skimage.morphology import opening

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
mask3D = spf.create_3D_mask(imdata, dims)

#OrthoSlicer3D(imdata).show()  # look at 3D volume data

# For slice position analysis want to do analysis on every slice but for now start with mid-slice
# TODO: only interested in slices 8 to 36 as this is where rods are... need to detect this range!!
# TODO: make this code work for every slice! Measurments need to be refined to match excel sheet.

error = []
distance = []
pos_m = []
pos_c = []
idx = 0  # need this to save first slice postion
too_many_regions = 0

for zz in np.linspace(7, 35, 29):  # TODO: change to range(no_slices):
    print('Slice ', int(zz+1))
    zz = int(zz)
    phmask = mask3D[zz, :, :]  # phantom mask
    phim = imdata[zz, :, :]*phmask  # phantom image
    bgim = imdata[zz, :, :]*~phmask  # background image

    ph_centre, pharea = spf.find_centre_and_area_of_phantom(phmask, plotflag=False)
    # TODO: use ^^ to help with detecting true phantom slices

    # display image
    # cv2.imshow('phantom image', ((phim/np.max(phim))*255).astype('uint8'))
    # cv2.waitKey(0)

    phim_dims = np.shape(phim)

    phim_norm = phim/np.max(phim)  # normalised image
    phim_gray = phim_norm*255  # greyscale image

    edged = cv2.Canny(phim_gray.astype('uint8'), 50, 150)  # edge detection
    bigbg = cv2.dilate(~phmask.astype('uint8'), None, iterations=4)  # dilate background mask
    bigbg[bigbg == 254] = 0

    edged = edged*~bigbg

    # cv2.imshow('Dilated Background', bigbg)
    # cv2.waitKey(0)

    # cv2.imshow('Canny Filter', edged.astype('float32'))
    # cv2.waitKey(0)

    edgedd = cv2.dilate(edged, None, iterations=1)

    # cv2.imshow('Canny Dilated', edgedd.astype('float32'))
    # cv2.waitKey(0)

    edgede = cv2.erode(edgedd, None, iterations=1)

    # cv2.imshow('Canny Eroded', edgede.astype('float32'))
    # cv2.waitKey(0)

    lines_im = phmask.copy()
    # LINE DETECTION
    minLineLength = 10
    maxLineGap = 10
    theta = np.pi/2  # 90 degrees to detect horizontal lines
    print(edgede.dtype, np.min(edgede), np.max(edgede))
    lines = cv2.HoughLinesP(edgede, 1, theta, 5, minLineLength, maxLineGap)

    no_lines = lines.shape
    no_lines = no_lines[0]
    #print('The number of lines detected is = ', no_lines)

    for lineno in np.linspace(0, no_lines-1, no_lines, dtype=int):
        for x1, y1, x2, y2 in lines[lineno]:
            cv2.line(lines_im, (x1, y1), (x2, y2), 0, 2)

    label_this = edgede*lines_im

    # cv2.imshow('After line removal', label_this.astype('float32'))
    # cv2.waitKey(0)

    label_this = opening(label_this)

    # cv2.imshow('After opening', label_this.astype('float32'))
    # cv2.waitKey(0)

    label_img, num = label(label_this, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
    print('Number of regions detected (should be 6!!!) = ', num)

    if num > 6:
        too_many_regions = too_many_regions + 1
        print('Too many regions detected!')

        label_this2 = label_this.copy()
        minLineLength = 2
        maxLineGap = 6
        theta = np.pi  # 0 degrees to detect vertical lines
        label_this2 = label_this2.astype('uint8')
        lines_im2 = phmask.copy()
        print(label_this2.dtype, np.min(label_this), np.max(label_this))
        lines = cv2.HoughLinesP(label_this2, 1, theta, 5, minLineLength, maxLineGap)

        no_lines = lines.shape
        no_lines = no_lines[0]
        # print('The number of lines detected is = ', no_lines)

        for lineno in np.linspace(0, no_lines - 1, no_lines, dtype=int):
            for x1, y1, x2, y2 in lines[lineno]:
                cv2.line(lines_im2, (x1, y1), (x2, y2), 0, 2)

        label_this3 = label_this2 * lines_im2

        # cv2.imshow('After 2nd line removal', label_this3.astype('float32'))
        # cv2.waitKey(0)

        label_img2, num2 = label(label_this3, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
        print('Number of regions detected (should be 6!!!) = ', num2)

        if num2 > 6:
            print('Still too many regions detected! =(')

        label_this = label_this3
        num = num2
        label_img = label_img2  # replace label_img with new version with less labelled regions

    # cv2.imshow('Rods labelled', label_img.astype('float32'))
    # cv2.waitKey(0)

    # TODO: put coloured rois over detected rods. Convert image to BGR.
    # plt.figure()
    # plt.imshow(label_img)
    # plt.show()

    props = regionprops(label_img)  # returns region properties for labelled image
    cent = np.zeros([num, 2])

    marker_im = phmask.copy()

    for xx in range(num):
        cent[xx, :] = props[xx].centroid  # central coordinate
        # TODO: or should this be centre of mass?

    cent = np.round(cent).astype(int)

    for i in cent:
        # draw the center of the circle
        cv2.circle(marker_im, (i[1], i[0 ]), 1, 0, 1)

    marker_im = marker_im*255

    # cv2.imshow('marker image', marker_im.astype('uint8'))
    # cv2.waitKey(0)

    """START MEASURING HERE"""

    temp1 = []  # top two rods
    temp2 = []  # middle, moving rods
    temp3 = []  # bottom two rods

    for i in cent:
        if i[0] < 80:
            temp1.append(i)
        if 100 < i[0] < 150:
            temp2.append(i)
        if i[0] > 180:
            temp3.append(i)

    # source need to be minimum x (left most rod in each pair)
    # dst needs to be maximum x (right most rod in each pair)

    # top 2 parallel rods
    indx = temp1[0][1] < temp1[1][1]
    if indx:
        src1 = temp1[0]  # (row, col) == (y, x)
        src1 = (src1[1], src1[0])  # (col, row) == (x, y)
        dst1 = temp1[1]
        dst1 = (dst1[1], dst1[0])
    elif not indx:
        src1 = temp1[1]  # (row, col) == (y, x)
        src1 = (src1[1], src1[0])  # (col, row) == (x, y)
        dst1 = temp1[0]
        dst1 = (dst1[1], dst1[0])

    # angled rods
    indx = temp2[0][1] < temp2[1][1]
    if indx:
        src2 = temp2[0]  # (row, col) == (y, x)
        src2 = (src2[1], src2[0])  # (col, row) == (x, y)
        dst2 = temp2[1]
        dst2 = (dst2[1], dst2[0])
    elif not indx:
        src2 = temp2[1]  # (row, col) == (y, x)
        src2 = (src2[1], src2[0])  # (col, row) == (x, y)
        dst2 = temp2[0]
        dst2 = (dst2[1], dst2[0])

    # bottom 2 parallel rods
    indx = temp3[0][1] < temp3[1][1]
    if indx:
        src3 = temp3[0]  # (row, col) == (y, x)
        src3 = (src3[1], src3[0])  # (col, row) == (x, y)
        dst3 = temp3[1]
        dst3 = (dst3[1], dst3[0])
    elif not indx:
        src3 = temp3[1]  # (row, col) == (y, x)
        src3 = (src3[1], src3[0])  # (col, row) == (x, y)
        dst3 = temp3[0]
        dst3 = (dst3[1], dst3[0])

    # TODO: replace marker_im with RGB/grayscale image of phantom and draw coloured lines on image instead
    hmarker_im = marker_im.copy()  # for horizontal lines
    vmarker_im = marker_im.copy()  # for vertical lines

    cv2.line(hmarker_im, src1, dst1, 0, 1)
    cv2.line(hmarker_im, src2, dst2, 0, 1)  # diagonal line between angled rods
    #cv2.line(hmarker_im, (src2[0], int(cent2[0, 0])), (dst2[0], int(cent2[0, 0])), 0, 1)  # horizontal lines between angled rods
    cv2.line(hmarker_im, src3, dst3, 0, 1)

    # cv2.imshow('horiz. marker image', hmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    cv2.line(vmarker_im, src1, src3, 0, 1)
    cv2.line(vmarker_im, dst1, dst3, 0, 1)
    #cv2.line(vmarker_im, (int(cent2[0, 1]), src2[1]), (int(cent2[0, 1]), dst2[1]), 0, 1)

    # cv2.imshow('vert. marker image', vmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    #compute the Euclidean distance between the midpoints
    # TODO: confirm what the voxel size is.
    # horizontal lines
    hdist1 = dist.euclidean(src1, dst1)
    hdist2 = dist.euclidean(src2, dst2)  # diagonal centre to centre
    #hdist2 = dist.euclidean((src2[0], int(cent2[0, 0])), (dst2[0], int(cent2[0, 0])))
    hdist3 = dist.euclidean(src3, dst3)

    #print('Horizontal distance (top) = ', hdist1, 'mm')
    #print('Horizontal distance between angled rods = ', hdist2, 'mm')
    #print('Measured distance between angled rods = ', hdist2, 'mm')
    #print('Horizontal distance (bottom) = ', hdist3, 'mm')

    # horizontal lines
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist1), (int(phim_dims[0]/2), int(dst1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist2), (src2[0] + 30, int(phim_dims[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist3), (int(phim_dims[0]/2), int(dst3[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    # cv2.imshow("horiz. measurements", hmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    # vertical lines
    # TODO: confirm voxel dimensions
    vdist1 = dist.euclidean(src1, src3)
    vdist2 = dist.euclidean(dst1, dst3)
    #vdist3 = dist.euclidean((int(cent2[0, 1]), src2[1]), (int(cent2[0, 1]), dst2[1]))

    #print('Vertical distance (left) = ', vdist1, 'mm')
    #print('Vertical distance (right) = ', vdist2, 'mm')
    #print('Vertical distance between angled rods = ', vdist3, 'mm')

    cv2.putText(vmarker_im, "{:.1f}mm".format(vdist1), (int(src1[0] + 5), int(src3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
    cv2.putText(vmarker_im, "{:.1f}mm".format(vdist2), (int(dst1[0] + 15), int(dst3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
    #cv2.putText(vmarker_im, "{:.1f}mm".format(vdist3), (int(phim_dims[0]/2), int(phim_dims[1]/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)

    # cv2.imshow("vert. measurements", vmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    # CALCULATIONS
    RDA = [120, 120, 120, 120]  # what the measurements should be
    RDM = [hdist1*0.98, hdist3*0.98, vdist1*0.98, vdist2*0.98]  # what is measured from the image
    # TODO: confirm voxel size is 1 x 1 mm or 0.98 x 0.98 mm

    CFall = np.divide(RDA, RDM)
    CF = np.mean(CFall)
    #print('Mean Correction Factor = ', CF)

    distance_of_angled_rods = hdist2*0.98  # distance between angled rods

    # TODO: remove brute force method with something more sophisticated and robust...
    if idx < 15:
        slice_position_measured = np.sqrt(((distance_of_angled_rods*CF)**2) - (6.5**2))/2  # how this relates to position
    elif idx >= 15:
        slice_position_measured = -np.sqrt(((distance_of_angled_rods*CF)**2) - (6.5**2))/2  # how this relates to position
    print('Measured position = ', slice_position_measured)

    if idx == 0:
        first_slice_position = slice_position_measured  # slice 1 get this from header info? is this slice number...?

    slice_position = idx  # slice number
    slice_thickness, slice_gap, pixeldims = spf.slice_pos_meta(ds)

    slice_position_calculated = first_slice_position - ((slice_position - 1)*slice_thickness)
    #print('Calculated position = ', slice_position_calculated)

    slice_position_error = slice_position_calculated - slice_position_measured
    #print('Error = ', slice_position_error, 'mm (must be within +- 1 mm)')

    distance.append(hdist2*0.98)
    pos_m.append(slice_position_measured)
    pos_c.append(slice_position_calculated)
    error.append(slice_position_error)

    idx = idx + 1

plt.figure()
plt.subplot(221)
plt.plot(np.linspace(8, 36, 29), distance)
plt.title('Slice Number vs. Distance')
plt.subplot(222)
plt.plot(np.linspace(8, 36, 29), pos_m)
plt.title('Slice Number vs. Measured Position')
plt.subplot(223)
plt.plot(np.linspace(8, 36, 29), pos_c)
plt.title('Slice Number vs. Calculated Position')
plt.subplot(224)
plt.plot(np.linspace(8, 36, 29), error)
plt.plot(np.linspace(8, 36, 29), np.repeat(-1, 29))
plt.plot(np.linspace(8, 36, 29), np.repeat(1, 29))
plt.title('Slice Number vs. Slice Position Error')
plt.show()


