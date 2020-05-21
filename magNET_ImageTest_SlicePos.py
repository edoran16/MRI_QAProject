"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

from DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
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
plt.figure()
plt.imshow(phim, cmap='bone')
plt.axis('off')
plt.show()

phim_dims = np.shape(phim)

phim_norm = phim/np.max(phim)
phim_gray = phim_norm*255

edged = cv2.Canny(phim_gray.astype('uint8'), 20, 200)
bigbg = cv2.dilate(~phmask.astype('uint8'), None, iterations=4)  # dilate background mask

edged = edged*~bigbg

plt.figure()
plt.subplot(141)
plt.imshow(bigbg)
plt.axis('off')
plt.subplot(142)
plt.imshow(edged)
plt.axis('off')

edgedd = cv2.dilate(edged, None, iterations=1)

plt.subplot(143)
plt.imshow(edgedd)
plt.axis('off')

edgede = cv2.erode(edgedd, None, iterations=1)

plt.subplot(144)
plt.imshow(edgede)
plt.axis('off')
plt.show()

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

plt.figure()
plt.subplot(121)
plt.imshow(label_this)
plt.axis('off')

label_img, num = label(label_this, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
print('Number of regions detected (should be 6!!!) = ', num)

plt.subplot(122)
plt.imshow(label_img)
plt.axis('off')
plt.show()

props = regionprops(label_img)  # returns region properties for labelled image
cent = np.zeros([num, 2])

marker_im = phmask.copy()

for xx in range(num):
    cent[xx, :] = props[xx].centroid  # central coordinate

cent = np.round(cent).astype(int)

for i in cent:
    # draw the center of the circle
    cv2.circle(marker_im, (i[0], i[1]), 1, 0, 1)

plt.figure()
plt.imshow(marker_im)
plt.axis('off')
plt.show()

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

src1 = temp1[0]
src1 = (src1[0], src1[1])
dst1 = temp1[1]
dst1 = (dst1[0], dst1[1])

src2 = temp2[0]
src2 = (src2[0], src2[1])
dst2 = temp2[1]
dst2 = (dst2[0], dst2[1])

src3 = temp3[0]
src3 = (src3[0], src3[1])
dst3 = temp3[1]
dst3 = (dst3[0], dst3[1])

cv2.line(marker_im, src1, dst1, 0, 1)
cv2.line(marker_im, src2, dst2, 0, 1)
cv2.line(marker_im, src3, dst3, 0, 1)

plt.figure()
plt.imshow(marker_im)
plt.axis('off')
plt.show()


# # find contours in the edge map
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
#
# # sort the contours from left-to-right and initialize the
# # 'pixels per metric' calibration variable
# (cnts, _) = contours.sort_contours(cnts)
# pixelsPerMetric = None
#
#
# # loop over the contours individually
# for c in cnts:
#     # if the contour is not sufficiently large, ignore it
#     if cv2.contourArea(c) >= 7:
#         continue
#     # compute the rotated bounding box of the contour
#     print(cv2.contourArea(c))
#     orig = phim_gray.copy()
#     box = cv2.minAreaRect(c)
#     box = cv2.boxPoints(box)
#     box = np.array(box, dtype="int")
#     # order the points in the contour such that they appear
#     # in top-left, top-right, bottom-right, and bottom-left
#     # order, then draw the outline of the rotated bounding
#     # box
#     box = perspective.order_points(box)
#     cv2.drawContours(orig, [box.astype("int")], -1, 0, 2)
#     # loop over the original points and draw them
#
#     for (x, y) in box:
#         cv2.circle(orig, (int(x), int(y)), 5, 0, -1)
#
#     cv2.imshow('orig', orig)
#     cv2.waitKey(0)
#
#     # # unpack the ordered bounding box, then compute the midpoint
#     # # between the top-left and top-right coordinates, followed by
#     # # the midpoint between bottom-left and bottom-right coordinates
#     # (tl, tr, br, bl) = box
#     # (tltrX, tltrY) = midpoint(tl, tr)
#     # (blbrX, blbrY) = midpoint(bl, br)
#     # # compute the midpoint between the top-left and top-right points,
#     # # followed by the midpoint between the top-righ and bottom-right
#     # (tlblX, tlblY) = midpoint(tl, bl)
#     # (trbrX, trbrY) = midpoint(tr, br)
#     # # draw the midpoints on the image
#     # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, 0, -1)
#     # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, 0, -1)
#     # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, 0, -1)
#     # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, 0, -1)
#     # # draw lines between the midpoints
#     # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), 0, 2)
#     # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), 0, 2)
#     #
#     # cv2.imshow('orig', orig)
#     # cv2.waitKey(0)
#     #
#     # # compute the Euclidean distance between the midpoints
#     # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#     # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#     # # if the pixels per metric has not been initialized, then
#     # # compute it as the ratio of pixels to supplied metric
#     # # (in this case, inches)
#     # if pixelsPerMetric is None:
#     #     pixelsPerMetric = dB / 120  # known width. Use 120 for now.
#     #
#     # # compute the size of the object
#     # dimA = dA / pixelsPerMetric
#     # dimB = dB / pixelsPerMetric
#     # # draw the object sizes on the image
#     # cv2.putText(orig, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
#     # cv2.putText(orig, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
#     # # show the output image
#     # cv2.imshow("Image", orig)
#     # cv2.waitKey(0)
#     #
#     # #print(dimA)
#     # #print(dimB)
#
