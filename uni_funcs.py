from pylab import *
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from DICOM_test import dicom_read_and_write
from nibabel.viewers import OrthoSlicer3D

from skimage.measure import profile_line, label, regionprops
import matplotlib.pyplot as plt
import numpy as np


def create_2D_mask(img, show_graphical=False):
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
    ch = ch.astype('uint8') * 255

    bin_ch = (ch / np.max(ch)).astype('uint8')  # binary mask [0, 1]

    if show_graphical:
        cv2.imshow('mask', ch)
        cv2.waitKey(0)

    return ch, bin_ch


def f_uni_meta(dicomfile):  # TODO: is this required??
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # rows and columns
    rows = dicomfile[0x0028, 0x0010]
    rows = rows.value
    cols = dicomfile[0x0028, 0x0011]
    cols = cols.value
    matrix_size = [rows, cols]

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # Per-frame Functional Groups Sequence
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        pixels_space = xx.PixelSpacing
        st = xx.SliceThickness

    # MR Averages Sequence
    elem5 = elem3.MRAveragesSequence
    for yy in elem5:
        NSA = yy.NumberOfAverages

    # (5200, 9229)  Shared Functional Groups Sequence
    elem6 = dicomfile[0x5200, 0x9229]
    seq2 = elem6.value
    elem7 = seq2[0]
    # print(elem7)
    elem8 = elem7.MRImagingModifierSequence
    for zz in elem8:
        PxlBW = zz.PixelBandwidth
        Tx_Freq = zz.TransmitterFrequency

    """ (0018, 9112) MR Timing and Related Parameters Sequence """

    elem9 = elem7.MRTimingAndRelatedParametersSequence
    for aa in elem9:
        TR = aa.RepetitionTime

    """ (0018, 9125) MR FOV / Geometry Sequence """
    elem10 = elem7.MRFOVGeometrySequence
    for bb in elem10:
        N_PE = bb[0x0018, 0x9231].value  # MRAcquisitionPhaseEncodingSteps

    return pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE


def check_ROI(roi_mask, phantom_image):  # TODO: get rid of
    # phantom_image is binary mask. Need to convert to greyscale.
    if np.max(phantom_image) == 1:  # binary
        phantom_image = phantom_image * 255

    # check ROI does not cover phantom i.e. covers any foreground signal
    # cv2.imshow('ROI mask', roi_mask)
    # cv2.waitKey(0)
    # cv2.imshow('Phantom Mask', phantom_image)
    # cv2.waitKey(0)

    sum_image = roi_mask + phantom_image
    sum_image = sum_image > 255
    sum_sum_image = np.sum(sum_image.astype('uint8'))

    if sum_sum_image > 0:
        print('Error with ROI placement!!! Overlap with phantom.')
        plt.figure()
        plt.imshow(sum_image)
        plt.show()

    # check ROI area has not extended beyond FOV
    roi_mask = roi_mask / np.max(roi_mask)  # convert to binary mask
    sum_roi_mask = np.sum(roi_mask)

    print('ROI area = ', sum_roi_mask, '(this should be 20 x 20 = 400)')

    if sum_roi_mask != 400:
        print('Error with ROI size! Matrix must extend beyond FOV.')

    if sum_sum_image == 0 and sum_roi_mask == 400:
        print('This ROI is perfectly fine.')


def sort_import_data(directpath, geometry, pt):
    with os.scandir(directpath) as the_folders:
        for folder in the_folders:
            fname = folder.name
            if re.search('-SNR_', fname):
                if re.search(geometry, fname):
                    if re.search(pt, fname):
                        if not re.search('_REPEAT', fname) and not re.search('_PR', fname) and not re.search('_OIL',fname):
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

                                # Sequence parameters required for normalised SNR calculation
                                pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = f_uni_meta(ds)

                                img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                                cv2.imshow('dicom imdata', img)
                                cv2.waitKey(0)

                            except ValueError:
                                print('DATA INPUT ERROR: this is 3D image data')
                                OrthoSlicer3D(imdata).show()  # look at 3D volume data
                                sys.exit()

    return img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE


def draw_centre_ROI(bin_mask, img, caseT, show_graphical=True):
    # TODO; does this affect SAG and COR views?
    # ACCOUNT FOR MISSING SIGNAL AT TOP OF PHANTOM (TRANSVERE VIEW ONLY).
    if caseT:
        oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
        oi[(img > filters.threshold_otsu(img)) == True] = 1  # Otsu threshold on image
        err = cv2.erode(oi, None, iterations=8)

        idx = np.where(err > 0)
        idx = idx[0]  # rows
        idx = idx[0]  # first row

        new_mask = bin_mask.copy()
        new_mask[0:idx, :] = 0
        mask = new_mask

    else:
        mask = bin_mask

        # cv2.imshow('mask', (bin_mask * 255))
        # cv2.waitKey(0)
        # cv2.imshow('otsu phantom', (1-oi)*255)
        # cv2.waitKey(0)
        # cv2.imshow('err', err * 255)
        # cv2.waitKey(0)
        # cv2.imshow('new mask', (new_mask * 255))
        # cv2.waitKey(0)

    # get centre of phantom and define ROI from there
    label_img, num = label(mask, connectivity=img.ndim, return_num=True)  # labels the mask

    props = regionprops(label_img, coordinates='rc')  # returns region properties for phantom mask ROI
    phantom_centre = props[0].centroid
    pc_row, pc_col = [int(phantom_centre[0]), int(phantom_centre[1])]

    # show detected regions and lines on marker_im
    marker_im = img.copy()
    marker_im = marker_im.astype('uint8')
    marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    cv2.line(marker_im, (pc_col + 10, pc_row + 10), (pc_col + 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col + 10, pc_row - 10), (pc_col - 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row - 10), (pc_col - 10, pc_row + 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row + 10), (pc_col + 10, pc_row + 10), (0, 0, 255), 1)

    area = ((pc_col + 10) - (pc_col - 10)) * ((pc_row + 10) - (pc_row - 10))
    print('Centre ROI Area =', area)
    area_aim = 20 * 20
    if area != area_aim:
        print('Signal ROI area is too large/too small')
        sys.exit()

    if show_graphical:
        cv2.imshow('Signal ROIs', marker_im)
        cv2.waitKey(0)

    return pc_row, pc_col, marker_im


def get_signal_value(imdata, pc_r, pc_c):
    # signal values corresponding to voxels inside each signal ROI (don't use greyscale image!)
    signal0 = np.mean(imdata[pc_r - 10:pc_r + 10, pc_c - 10:pc_c + 10])

    # test = imdata.copy()
    # test[pc_row - 10:pc_row + 10, pc_col - 10:pc_col + 10] = 0
    # plt.figure()
    # plt.imshow(test)
    # plt.show()

    print('Mean signal (total) =', signal0)

    return signal0


def obtain_uniformity_profile(imdata, src, dst, caseH, caseV, show_graphical=False):
    # src and dst are tuples of (x, y) i.e. (column, row)

    # draw line profile across centre line of phantom
    outputs = []
    improfile = np.copy(imdata)
    improfile = (improfile / np.max(improfile))  # normalised
    improfile = improfile * 255  # greyscale
    improfile = improfile.astype('uint8')
    improfile = cv2.cvtColor(improfile, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    # cv2.imshow('test', improfile)

    for xx in np.linspace(-4, 5, 10):
        if caseH:  # horizontal lines
            print('HORIZONTAL PROFILE')  # drawn top of image to bottom of image
            src2 = (src[0], int(src[1] + xx))  # starting point (x, y)
            dst2 = (dst[0], int(dst[1] + xx))  # finish point
        if caseV:  # vertical lines
            print('VERTICAL PROFILE')  # drawn LHS to RHS of image
            src2 = (int(src[0] + xx), int(src[1]))  # starting point
            dst2 = (int(dst[0] + xx), int(dst[1]))  # finish point
        #
        # test = imdata.copy()
        # test[src2[1], src2[0]] = 15000
        # test[dst2[1], dst2[0]] = 25000
        # plt.figure()
        # plt.imshow(test)
        # plt.show()

        output = profile_line(imdata, src2, dst2)  # voxel values along specified line, coords specified (x, y)
        outputs.append(output)
        if xx == 0:
            print('centre line!')
            improfile = display_profile_line(improfile, src2, dst2, caseH, caseV, linecolour=(0, 0, 255), show_graphical=False)
        else:
            improfile = display_profile_line(improfile, src2, dst2, caseH, caseV, linecolour=(255, 0, 0), show_graphical=False)

    cv2.imshow('Individual Profile Line', improfile)
    cv2.waitKey(0)

    mean_output = np.mean(outputs, axis=0)

    # plot profile line outputs + mean output vs. voxels sampled
    if show_graphical:
        plt.figure()
        plt.subplot(221)
        for ee in range(10):
            plt.plot(outputs[ee], 'b')
        plt.plot(mean_output, 'r')
        plt.xlabel('Voxels')
        plt.ylabel('Signal')
        plt.show()

    return mean_output


def display_profile_line(imdata, src, dst, caseH, caseV, linecolour, show_graphical=False):
    # display profile line on phantom: from source code of profile_line function
    src_col, src_row = np.asarray(src, dtype=float)  # src = (x, y) = (col, row)
    dst_col, dst_row = np.asarray(dst, dtype=float)

    dims = np.shape(imdata)

    if caseH:
        rows = np.repeat(int(src_row), dims[0])
        cols = np.linspace(int(src_col-1), int(dst_col-1), dims[1])
        # TODO: SPEC
    if caseV:
        rows = np.linspace(int(src_row-1), int(dst_row-1), dims[0])
        cols = np.repeat(int(src_col), dims[1])
        # TODO: SPEC

    imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)] = linecolour

    imdata[np.array([spec_rows], dtype=int), np.array([spec_cols], dtype=int)] = (0, 255, 0)

    # plot sampled line on phantom to visualise where output comes from
    if show_graphical:
        cv2.imshow('Individual Profile Line', imdata)
        cv2.waitKey(0)

    return imdata


def calc_fUniformity(signal, uniformity_range):
    total_no_of_voxels = len(signal)
    no_voxels_in_range = 0
    for dd in range(total_no_of_voxels):
        if uniformity_range[0] <= signal[dd] <= uniformity_range[1]:
            no_voxels_in_range = no_voxels_in_range + 1

    fractional_uniformity = no_voxels_in_range / total_no_of_voxels

    return fractional_uniformity



