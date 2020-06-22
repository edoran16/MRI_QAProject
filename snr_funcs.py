
from pylab import *
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_2D_mask(img):
    """ input:  img is  greyscale uint8 image data from DICOM
        output: ch is 2D mask (also grayscale!!!!)"""

    h = ex.equalize_hist(img)  # histogram equalisation increases contrast of image

    oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
    oi[(img > filters.threshold_otsu(img)) == True] = 255  # Otsu threshold on image

    oh = np.zeros_like(img, dtype=np.uint8)  # zero array same dims as img
    oh[(h > filters.threshold_otsu(h)) == True] = 255  # Otsu threshold on hist eq image

    nm = img.shape[0] * img.shape[1]  # total number of voxels in image
    # calculate normalised weights for weighted combination
    w1 = np.sum(oi) / nm
    w2 = np.sum(oh) / nm
    ots = np.zeros_like(img, dtype=np.uint8)  # create final zero array
    new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
    ots[(new > filters.threshold_otsu(new)) == True] = 255  # Otsu threshold on weighted combination

    # cv2.imshow('ots', ots)
    # cv2.waitKey(0)

    eroded_ots = cv2.erode(ots, None, iterations=3)
    dilated_ots = cv2.dilate(eroded_ots, None, iterations=3)
    #
    # cv2.imshow('dilated', dilated_ots)
    # cv2.waitKey(0)

    openhull = opening(dilated_ots)

    # cv2.imshow('openhull', openhull)
    # cv2.waitKey(0)

    conv_hull = convex_hull_image(openhull)

    ch = np.multiply(conv_hull, 1)  # bool --> binary
    ch = ch.astype('uint8')*255

    return ch


def snr_meta(dicomfile):
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # rows and columns
    rows = dicomfile[0x0028, 0x0010]
    rows = rows.value
    cols = dicomfile[0x0028, 0x0011]
    cols = cols.value
    matrix_size = [rows, cols]

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # pydicom.dataelem.DataElement
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        pixels_space = xx.PixelSpacing

    return matrix_size, pixels_space


def check_ROI(roi_mask, phantom_image):
    # phantom_image is binary mask. Need to convert to greyscale.
    if np.max(phantom_image) == 1:  # binary
        phantom_image = phantom_image * 255

    # check ROI does not cover phantom i.e. covers any foreground signal
    # cv2.imshow('ROI mask', roi_mask)
    # cv2.waitKey(0)
    # cv2.imshow('Phantom Mask', phantom_image)
    # cv2.waitKey(0)

    sum_image = roi_mask+phantom_image
    sum_image = sum_image > 255
    sum_sum_image = np.sum(sum_image.astype('uint8'))

    if sum_sum_image > 0:
        print('Error with ROI placement!!! Overlap with phantom.')
        plt.figure()
        plt.imshow(sum_image)
        plt.show()

    # check ROI area has not extended beyond FOV
    roi_mask = roi_mask/np.max(roi_mask)  # convert to binary mask
    sum_roi_mask = np.sum(roi_mask)

    print('ROI area = ', sum_roi_mask, '(this should be 20 x 20 = 400)')

    if sum_roi_mask != 400:
        print('Error with ROI size! Matrix must extend beyond FOV.')

    if sum_sum_image == 0 and sum_roi_mask == 400:
        print('This ROI is perfectly fine.')


