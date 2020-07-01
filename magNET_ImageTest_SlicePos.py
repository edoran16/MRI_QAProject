"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import slice_pos_funcs as spf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
from skimage import filters

from DICOM_test import dicom_read_and_write
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!
from skimage.morphology import opening

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/Slice_Position_Images/"
folder = "42-SLICE_POS"

pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

with os.scandir(pathtodicom) as it:
    for file in it:
        path = "{0}{1}".format(pathtodicom, file.name)

ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

try:
    xdim, ydim, zdim = dims
    # OrthoSlicer3D(imdata).show()  # look at 3D volume data
except ValueError:
    print('DATA INPUT ERROR: this is 2D image data')
    sys.exit()

# create 3D mask
mask3D, no_slices = spf.create_3D_mask(imdata, dims)

# OrthoSlicer3D(imdata).show()  # look at 3D volume data

# initialise variables for plotting results
error = []
distance = []
pos_m = []
pos_c = []
idx = 0  # need this to save first slice position
too_many_regions = 0  # to improve rod detection error
switch_sign = False  # INITIALLY FALSE: for switching plus/minus sign in equation for measured slice position
show_meas = False  # for showing measurements made on each slice iteration
# for video
img_array = []
make_video = False  # produce 2 videos
# for calculations
first_slice_position = []

# detect slice range for analysis
start_slice, last_slice, pf_img_array = spf.find_range_slice_pos(no_slices, mask3D, imdata, plotflag=False, savepng=False)

show_graphical = False  # display pre-processing steps
show_graphical2 = True  # replacing erosion/dilation method with otsu threshold
otsu_method = True  # use otsu method for image pre-processing

for zz in range(start_slice, last_slice+1):
    print(zz)
    zz = int(zz)
    print('Actual Slice Number ', zz+1)
    print('Relevant Slice Number', idx+1)

    phmask = mask3D[zz, :, :]  # phantom mask
    phim = imdata[zz, :, :]*phmask  # phantom image
    bgim = imdata[zz, :, :]*~phmask  # background image

    ph_centre, pharea = spf.find_centre_and_area_of_phantom(phmask, plotflag=False)
    # use ph_centre for defining where to put measurement text on final display

    # display image
    if show_graphical:
        cv2.imwrite("{0}phantom_image_slice_{1}.png".format(imagepath, zz+1), ((phim/np.max(phim))*255).astype('uint8'))
        cv2.imshow('phantom image', ((phim/np.max(phim))*255).astype('uint8'))
        cv2.waitKey(0)

    phim_dims = np.shape(phim)

    phim_norm = phim/np.max(phim)  # normalised image
    phim_gray = phim_norm*255  # greyscale image

    edged = cv2.Canny(phim_gray.astype('uint8'), 50, 150)  # edge detection
    bigbg = cv2.dilate(~phmask.astype('uint8'), None, iterations=4)  # dilate background mask
    bigbg[bigbg == 254] = 0

    edged = edged*~bigbg

    if show_graphical:
        cv2.imwrite("{0}dilated_background_slice_{1}.png".format(imagepath, zz + 1), bigbg)
        cv2.imshow('Dilated Background', bigbg)
        cv2.waitKey(0)

        cv2.imwrite("{0}cannyfilter1_slice_{1}.png".format(imagepath, zz + 1), (edged*255).astype('uint8'))
        cv2.imshow('Canny Filter', edged.astype('float32'))
        cv2.waitKey(0)

    edgedd = cv2.dilate(edged, None, iterations=1)

    if show_graphical:
        cv2.imwrite("{0}dilation1_slice_{1}.png".format(imagepath, zz + 1), edgedd.astype('float32'))
        cv2.imshow('Canny Dilated', edgedd.astype('float32'))
        cv2.waitKey(0)

    edgede = cv2.erode(edgedd, None, iterations=1)

    if show_graphical2:
        cv2.imwrite("{0}erosion1_slice_{1}.png".format(imagepath, zz + 1), (edgede*255).astype('uint8'))
        cv2.imshow('Canny Eroded', edgede.astype('float32'))
        cv2.waitKey(0)

    # OTSU METHOD COMPARISON
    ots = np.zeros_like(phim_gray, dtype=np.uint16)  # creates zero array same dimensions as img
    ots[(phim_gray > filters.threshold_otsu(phim_gray)) == True] = 1  # Otsu threshold on image

    if show_graphical2:
        cv2.imwrite("{0}otsuthresh_slice_{1}.png".format(imagepath, zz + 1), (ots*255).astype('uint8'))
        cv2.imshow('INVERSE OTS FOR COMP', ((1 - ots)*~bigbg).astype('float32'))
        cv2.waitKey(0)

    if otsu_method:
        remove_lines = (((1 - ots)*~bigbg)/np.max((1 - ots)*~bigbg)).astype('uint8')  # otsu threshold method
        # print(remove_lines.dtype, np.min(remove_lines), np.max(remove_lines))
    else:
        remove_lines = edgede  # edge/dilation method
        print(remove_lines.dtype, np.min(remove_lines), np.max(remove_lines))

    lines_im = phmask.copy()
    # LINE DETECTION
    minLineLength = 10
    maxLineGap = 10
    theta = np.pi/2  # 90 degrees to detect horizontal lines
    lines = cv2.HoughLinesP(remove_lines, 1, theta, 5, minLineLength, maxLineGap)

    no_lines = lines.shape
    no_lines = no_lines[0]

    for lineno in np.linspace(0, no_lines-1, no_lines, dtype=int):
        for x1, y1, x2, y2 in lines[lineno]:
            cv2.line(lines_im, (x1, y1), (x2, y2), 0, 2)

    label_this = remove_lines*lines_im

    if show_graphical:
        cv2.imwrite("{0}lineremoval1_slice_{1}.png".format(imagepath, zz + 1), (label_this*255).astype('uint8'))
        cv2.imshow('After line removal', label_this.astype('float32'))
        cv2.waitKey(0)

    if not otsu_method:
        label_this = opening(label_this)

        if show_graphical:
            cv2.imwrite("{0}opening1_slice_{1}.png".format(imagepath, zz + 1), (label_this*255).astype('uint8'))
            cv2.imshow('After opening', label_this.astype('float32'))
            cv2.waitKey(0)

    label_img, num = label(label_this, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
    print('Number of regions detected (should be 6!!!) = ', num)

    if num > 6:
        too_many_regions = too_many_regions + 1
        print('Too many regions detected! =O')

        label_this2 = label_this.copy()
        minLineLength = 2
        maxLineGap = 6
        theta = np.pi  # 0 degrees to detect vertical lines
        label_this2 = label_this2.astype('uint8')
        lines_im2 = phmask.copy()
        lines = cv2.HoughLinesP(label_this2, 1, theta, 5, minLineLength, maxLineGap)

        no_lines = lines.shape
        no_lines = no_lines[0]

        for lineno in np.linspace(0, no_lines - 1, no_lines, dtype=int):
            for x1, y1, x2, y2 in lines[lineno]:
                cv2.line(lines_im2, (x1, y1), (x2, y2), 0, 2)

        label_this3 = label_this2 * lines_im2

        if show_graphical:
            cv2.imwrite("{0}lineremoval2_slice_{1}.png".format(imagepath, zz + 1), (label_this3*255).astype('uint8'))
            cv2.imshow('After 2nd line removal', label_this3.astype('float32'))
            cv2.waitKey(0)

        label_img2, num2 = label(label_this3, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
        print('Number of regions detected (should be 6!!!) = ', num2)

        if num2 > 6:
            ValueError('Too many regions detected! =(')

        label_this = label_this3
        num = num2
        label_img = label_img2  # replace label_img with new version with less labelled regions

    if show_graphical:
        cv2.imwrite("{0}labelled_rods_slice_{1}.png".format(imagepath, zz + 1), (label_img*255).astype('uint8'))
        cv2.imshow('Rods labelled', label_img.astype('float32'))
        cv2.waitKey(0)

    props = regionprops(label_img)  # returns region properties for labelled image
    cent = np.zeros([num, 2])

    marker_im = phim_gray.copy()
    marker_im = marker_im.astype('uint8')
    marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    for xx in range(num):
        cent[xx, :] = props[xx].centroid  # central coordinate

    cent = np.round(cent).astype(int)  # TODO: resave all figures with this amendment!!

    for i in cent:
        # draw the center of the circle
        cv2.circle(marker_im, (i[1], i[0]), 1, (0, 0, 255), 1)

    if show_graphical:
        cv2.imwrite("{0}marker_image_slice_{1}.png".format(imagepath, zz + 1), marker_im.astype('uint8'))
        cv2.imshow('marker image', marker_im.astype('uint8'))
        cv2.waitKey(0)

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

    hmarker_im = phim_gray.copy()
    hmarker_im = hmarker_im.astype('uint8')
    hmarker_im = cv2.cvtColor(hmarker_im, cv2.COLOR_GRAY2BGR)  # for horizontal lines

    vmarker_im = phim_gray.copy()
    vmarker_im = vmarker_im.astype('uint8')
    vmarker_im = cv2.cvtColor(vmarker_im, cv2.COLOR_GRAY2BGR)  # for horizontal lines

    cv2.line(hmarker_im, src1, dst1, (0, 0, 255), 1)
    cv2.line(hmarker_im, src2, dst2, (0, 0, 255), 1)  # diagonal line between angled rods
    cv2.line(hmarker_im, src3, dst3, (0, 0, 255), 1)

    # cv2.imshow('horiz. marker image', hmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    cv2.line(vmarker_im, src1, src3, (0, 0, 255), 1)
    cv2.line(vmarker_im, dst1, dst3, (0, 0, 255), 1)

    # cv2.imshow('vert. marker image', vmarker_im.astype('uint8'))
    # cv2.waitKey(0)

    # compute the Euclidean distance between the midpoints (output in terms of number of voxels)
    # horizontal lines
    hdist1 = dist.euclidean(src1, dst1)
    hdist2 = dist.euclidean(src2, dst2)  # diagonal centre to centre
    hdist3 = dist.euclidean(src3, dst3)

    # horizontal lines
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist1), (int(phim_dims[0]/2), int(dst1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist2), (src2[0] + 30, int(phim_dims[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdist3), (int(phim_dims[0]/2), int(dst3[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # vertical lines
    vdist1 = dist.euclidean(src1, src3)  # LHS
    vdist2 = dist.euclidean(dst1, dst3)  # RHS

    cv2.putText(vmarker_im, "{:.1f}mm".format(vdist1), (int(src1[0] + 5), int(src3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(vmarker_im, "{:.1f}mm".format(vdist2), (int(dst1[0] + 15), int(dst3[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    final_markers = cv2.vconcat((hmarker_im.astype('uint8'), vmarker_im.astype('uint8')))

    if show_meas:
        cv2.imwrite("{0}final_measures_slice_{1}.png".format(imagepath, zz + 1), final_markers)
        cv2.imshow("Measurements", final_markers)
        cv2.waitKey(0)

    # CALCULATIONS
    RDA = [120, 120, 120, 120]  # what the measurements should be

    matrix_size, slice_thickness, pixeldims = spf.slice_pos_meta(ds)

    RDM = [hdist1/pixeldims[0], hdist3/pixeldims[0], vdist1/pixeldims[1], vdist2/pixeldims[1]]
    # what is measured from the image slice ^^^

    CFall = np.divide(RDA, RDM)
    CF = np.mean(CFall)  # average of 4 measurements

    print(pixeldims)
    if pixeldims[0] == pixeldims[1]:
        pixeldim = pixeldims[0]
    else:
        ValueError('NEED TO ACCOUNT FOR ANISOTROPIC VOXELS')

    distance_of_angled_rods = hdist2/pixeldim  # distance between angled rods

    # detect middle slice where rods are vertically in line - where plus and minus switch in calculation
    if 4.5 < distance_of_angled_rods < 8.5:  # should be 6.5 mm with tolerance of +/- 2 mm
        switch_sign = True
        print('THE SIGN IS SWITCHED!!!!')

    # plus sign
    slice_position_measured = np.sqrt(((distance_of_angled_rods*CF)**2) - (6.5**2))/2
    # how measurement relates to slice position

    # minus sign
    if switch_sign:
        slice_position_measured = -np.sqrt(((distance_of_angled_rods*CF)**2) - (6.5**2))/2
        # how measurement relates to slice position

    if idx == 0:
        first_slice_position = slice_position_measured
        # first calculated slice position = first measured slice position
    else:
        first_slice_position = first_slice_position

    slice_position = idx+1  # slice number

    slice_position_calculated = first_slice_position - ((slice_position - 1)*slice_thickness)

    slice_position_error = slice_position_calculated - slice_position_measured

    img_array.append(final_markers)  # for making video
    vid_dims = final_markers.shape  # for making video
    # for plotting
    distance.append(hdist2)
    pos_m.append(slice_position_measured)
    pos_c.append(slice_position_calculated)
    error.append(slice_position_error)

    idx = idx + 1  # index for next slice

slices_vec = np.linspace(start_slice+1, last_slice+1, idx)

plt.figure()
plt.subplot(131)
plt.plot(slices_vec, distance)
plt.xlim([1, no_slices])
plt.title('Slice Number vs. Distance')
plt.subplot(132)
plt.plot(slices_vec, pos_m)
plt.plot(slices_vec, pos_c)
plt.xlim([1, no_slices])
plt.legend(['Measured', 'Calculated'])
plt.title('Slice Number vs. Position')
plt.subplot(133)
plt.plot(slices_vec, error)
plt.plot(slices_vec, np.repeat(-2, idx), 'r')
plt.plot(slices_vec, np.repeat(2, idx), 'r')
plt.xlim([1, no_slices])
plt.title('Slice Number vs. Slice Position Error')
plt.show()

# mean error
mean_error = np.mean(error)
print('The mean slice position error = ', mean_error.round(2))  # -0.73
# standard deviation error
stdev_error = np.std(error)
print('The standard deviation of the error = ', stdev_error.round(2))  # 0.31
# range of error
range_error = np.max(error) - np.min(error)  # 1.52 range from -1.01 to 0.51
print('The range of the slice position error is = ', range_error.round(2), 'ranging from', (np.min(error)).round(2), 'to', (np.max(error)).round(2))

if make_video:
    # create video of pass/fail assignment
    spf.make_video_from_img_array(pf_img_array, (phim_dims[1], phim_dims[0]), 'SlicePos_findrange.mp4')
    # create video of measurements
    spf.make_video_from_img_array(img_array, (vid_dims[1], vid_dims[0]), 'SlicePos.mp4')

# Comparison with MagNET Report
df = pd.read_excel(r'Sola_INS_07_05_19.xls', sheet_name='Slice Position Sola (2)')

sola_distance = df.Position  # slice 7 -> 36 (I analyse slice 8 to 36)
sola_distance = sola_distance[1:30]

sola_pos_m = df['Unnamed: 5']
sola_pos_m = sola_pos_m[1:30]

sola_pos_c = df['Unnamed: 6']
sola_pos_c = sola_pos_c[1:30]

sola_error = df['Unnamed: 7']
sola_error = sola_error[1:30]

plt.figure()
plt.subplot(221)
plt.plot(slices_vec, distance)
plt.plot(slices_vec, sola_distance)
plt.xlim([1, no_slices])
plt.legend(['Python', 'Macro'])
plt.title('Measured Rod Distance')

plt.subplot(222)
plt.plot(slices_vec, pos_m)
plt.plot(slices_vec, sola_pos_m)
plt.xlim([1, no_slices])
plt.legend(['Python', 'Macro'])
plt.title('Measured Position')

plt.subplot(223)
plt.plot(slices_vec, pos_c)
plt.plot(slices_vec, sola_pos_c)
plt.xlim([1, no_slices])
plt.legend(['Python', 'Macro'])
plt.title('Calculated Position')

plt.subplot(224)
plt.plot(slices_vec, error)
plt.plot(slices_vec, sola_error)
plt.plot(slices_vec, np.repeat(-2, len(slices_vec)), 'r')
plt.plot(slices_vec, np.repeat(2, len(slices_vec)), 'r')
plt.plot(slices_vec, np.repeat(0, len(slices_vec)), 'r--')
plt.xlim([1, no_slices])
plt.legend(['Python', 'Macro', 'Pass/Fail Region'], loc='upper right', bbox_to_anchor=(1, 0.9), fontsize='x-small')
plt.title('Position Error')
plt.show()


