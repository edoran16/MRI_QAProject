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


def f_uni_meta(dicomfile):
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # Per-frame Functional Groups Sequence
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        pixels_space = xx.PixelSpacing

    return pixels_space


def sort_import_data(directpath, geometry, pt, show_graphical=False, imagepath=None):
    with os.scandir(directpath) as the_folders:
        for folder in the_folders:
            fname = folder.name
            if re.search('-SNR_', fname):
                if re.search(geometry, fname):
                    if re.search(pt, fname):
                        if not re.search('_REPEAT', fname) and not re.search('_PR', fname) and not re.search('_OIL',fname):
                            print('Loading ...', fname)

                            ## DOCKER/XNAT DEVELOPMENT
                            # Only run this analysis if SNR is in the scan name
                            x = re.search('SNR', fname)
                            if x:
                                y = 1
                            try:
                                print(y)
                                print('This scan WAS acquired for SNR and Uniformity tests.')
                            except:
                                print('This scan WAS NOT acquired for SNR and Uniformity test.')
                                exit(1)  # exit code
                            ########################

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
                                pixels_space = f_uni_meta(ds)

                                img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                                if show_graphical:
                                    cv2.imwrite("{0}flood_image.png".format(imagepath), img)
                                    cv2.imshow('dicom imdata', img)
                                    cv2.waitKey(0)

                            except ValueError:
                                print('DATA INPUT ERROR: this is 3D image data')
                                OrthoSlicer3D(imdata).show()  # look at 3D volume data
                                sys.exit()

    return img, imdata, pixels_space


def draw_centre_ROI(bin_mask, img, caseT, show_graphical=True, imagepath=None):
    # ACCOUNT FOR MISSING SIGNAL AT TOP OF PHANTOM (TRANSVERSE VIEW ONLY).
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

    dims = np.shape(img)
    # cv2.line src/dst defined x, y same orientation and row, col. [0,0] = top left corner
    # PROOF OF CONCEPT LINES
    # cv2.line(marker_im, (10, 41), (246, 41), (255, 0, 255), 1)
    # cv2.line(marker_im, (10, 215), (246, 215), (255, 0, 255), 1)

    cv2.line(marker_im, (pc_col + 10, pc_row + 10), (pc_col + 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col + 10, pc_row - 10), (pc_col - 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row - 10), (pc_col - 10, pc_row + 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row + 10), (pc_col + 10, pc_row + 10), (0, 0, 255), 1)

    area = ((pc_col + 10) - (pc_col - 10)) * ((pc_row + 10) - (pc_row - 10))
    # print('Centre ROI Area =', area)
    area_aim = 20 * 20
    if area != area_aim:
        print('Signal ROI area is too large/too small')
        sys.exit()

    if show_graphical:
        cv2.imwrite("{0}centre_ROI_image.png".format(imagepath), marker_im)
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


def obtain_uniformity_profile(imdata, src, dst, pc_row, pc_col, dist80, caseH, caseV, show_graphical=False, imagepath=None):
    # src and dst are tuples of (x, y) i.e. (column, row)
    # draw line profile across centre line of phantom
    outputs = []
    improfile = np.copy(imdata)
    improfile = (improfile / np.max(improfile))  # normalised
    improfile = improfile * 255  # greyscale
    improfile = improfile.astype('uint8')
    improfile = cv2.cvtColor(improfile, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    dims = np.shape(imdata)

    for xx in np.linspace(-4, 5, 10):
        if caseH:  # horizontal lines
            # print('HORIZONTAL PROFILE')  # drawn top of image to bottom of image
            src2 = (src[0], int(src[1] + xx))  # starting point (x, y)
            dst2 = (dst[0], int(dst[1] + xx))  # finish point
            # to get line profile output
            rows = np.repeat(src2[1], dims[0])
            cols = np.linspace(src2[0], dst2[0], dims[1])
        if caseV:  # vertical lines
            # print('VERTICAL PROFILE')  # drawn LHS to RHS of image
            src2 = (int(src[0] + xx), int(src[1]))  # starting point
            dst2 = (int(dst[0] + xx), int(dst[1]))  # finish point
            # to get line profile output
            rows = np.linspace(src2[1], dst2[1], dims[0])
            cols = np.repeat(src2[0], dims[1])

        output = imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)]
        outputs.append(output)

        if xx == 0:
            # print('centre line!')
            improfile = display_profile_line(improfile, src2, dst2, pc_row, pc_col, dist80, caseH, caseV, linecolour=(0, 0, 255), show_graphical=False)
        else:
            improfile = display_profile_line(improfile, src2, dst2, pc_row, pc_col, dist80, caseH, caseV, linecolour=(255, 0, 0), show_graphical=False)

        if caseH:
            cv2.imwrite("{0}profile_line_imageH.png".format(imagepath), improfile)
        if caseV:
            cv2.imwrite("{0}profile_line_imageV.png".format(imagepath), improfile)

        if show_graphical:
            cv2.imshow('Individual Profile Line', improfile)
            cv2.waitKey(0)

    mean_output = np.mean(outputs, axis=0)

    # plot profile line outputs + mean output vs. voxels sampled
    if show_graphical:
        plt.figure()
        if dist80 != 0:
            plt.subplot(221)
        for ee in range(10):
            plt.plot(outputs[ee], 'b')
        plt.plot(mean_output, 'r')
        plt.xlabel('Pixel Number')
        plt.ylabel('Signal')
        plt.show()

    return mean_output


def display_profile_line(imdata, src, dst, pc_row, pc_col, dist80, caseH, caseV, linecolour, show_graphical=False, imagepath=None):
    # display profile line on phantom: from source code of profile_line function
    src_col, src_row = np.asarray(src, dtype=float)  # src = (x, y) = (col, row)
    dst_col, dst_row = np.asarray(dst, dtype=float)

    dims = np.shape(imdata)

    if caseH:
        rows = np.repeat(int(src_row), dims[0])
        cols = np.linspace(int(src_col-1), int(dst_col-1), dims[1])
        # Add 160 mm lines dist80mm = 82
        # spec_rows = np.linspace(0, dims[0]-1, dims[0])
        # spec_cols = np.repeat(pc_col - dist80, dims[1])
        # spec_rows2 = np.linspace(0, dims[0] - 1, dims[0])
        # spec_cols2 = np.repeat(pc_col + dist80, dims[1])

    if caseV:
        rows = np.linspace(int(src_row-1), int(dst_row-1), dims[0])
        cols = np.repeat(int(src_col), dims[1])
        # Add 160 mm lines dist80mm = 82
        # spec_rows = np.repeat(pc_row - dist80, dims[0])
        # spec_cols = np.linspace(0, dims[1] - 1, dims[1])
        # spec_rows2 = np.repeat(pc_row + dist80, dims[0])
        # spec_cols2 = np.linspace(0, dims[1] - 1, dims[1])

    imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)] = linecolour

    # 160 mm regions
    if dist80 != 0:
        if caseH:
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col + dist80, pc_row), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col - dist80, pc_row), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.putText(imdata, "160 mm", (pc_col-20, pc_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if caseV:
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col, pc_row + dist80), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col, pc_row - dist80), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.putText(imdata, "160 mm", (pc_col + 10, pc_row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # imdata[np.array([spec_rows], dtype=int), np.array([spec_cols], dtype=int)] = (0, 255, 0)
    # imdata[np.array([spec_rows2], dtype=int), np.array([spec_cols2], dtype=int)] = (0, 255, 0)

    # plot sampled line on phantom to visualise where output comes from
    if show_graphical:
        cv2.imwrite("{0}just_another_profile_line_image.png".format(imagepath), imdata)
        cv2.imshow('Individual Profile Line!!', imdata)
        cv2.waitKey(0)

    return imdata


def calc_fUniformity(signal, uniformity_range):
    total_no_of_voxels = len(signal)
    no_voxels_in_range = 0
    for dd in range(total_no_of_voxels):
        if uniformity_range[0] <= signal[dd] <= uniformity_range[1]:
            no_voxels_in_range = no_voxels_in_range + 1

    fractional_uniformity = no_voxels_in_range / total_no_of_voxels
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)

    return fractional_uniformity, mean_signal, std_signal



