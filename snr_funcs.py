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


def create_2D_mask(img, show_graphical=False, imagepath=None):
    """ input:  img is  greyscale uint8 image data from DICOM
        imagepath = where to save png
        output: ch is 2D mask (also grayscale!!!!)"""

    if show_graphical:
        cv2.imwrite("{0}img.png".format(imagepath), img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    h = ex.equalize_hist(img)  # histogram equalisation increases contrast of image
    hn = h - np.min(h)
    hn2 = (hn / np.max(hn)) * 255
    print(np.max(hn2), np.min(hn2))

    if show_graphical:
        cv2.imwrite("{0}h.png".format(imagepath), hn2.astype('uint8'))
        cv2.imshow('h', h)
        cv2.waitKey(0)

    oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
    oi[(img > filters.threshold_otsu(img)) == True] = 255  # Otsu threshold on image

    if show_graphical:
        cv2.imwrite("{0}oi.png".format(imagepath), oi)
        cv2.imshow('oi', oi)
        cv2.waitKey(0)

    oh = np.zeros_like(img, dtype=np.uint8)  # zero array same dims as img
    oh[(h > filters.threshold_otsu(h)) == True] = 255  # Otsu threshold on hist eq image

    if show_graphical:
        cv2.imwrite("{0}oh.png".format(imagepath), oh)
        cv2.imshow('oh', oh)
        cv2.waitKey(0)

    nm = img.shape[0] * img.shape[1]  # total number of voxels in image
    # calculate normalised weights for weighted combination
    w1 = np.sum(oi) / nm
    w2 = np.sum(oh) / nm
    ots = np.zeros_like(img, dtype=np.uint8)  # create final zero array
    new = ((w1 * img) + (w2 * h)) / (w1 + w2)  # weighted combination of original image and hist eq version
    newn = new - np.min(new)
    newn2 = (newn / np.max(newn)) * 255

    if show_graphical:
        cv2.imwrite("{0}new.png".format(imagepath), newn2.astype('uint8'))
        cv2.imshow('new', new)
        cv2.waitKey(0)

    ots[(new > filters.threshold_otsu(new)) == True] = 255  # Otsu threshold on weighted combination

    if show_graphical:
        cv2.imwrite("{0}ots.png".format(imagepath), ots)
        cv2.imshow('ots', ots)
        cv2.waitKey(0)

    eroded_ots = cv2.erode(ots, None, iterations=3)
    dilated_ots = cv2.dilate(eroded_ots, None, iterations=3)

    if show_graphical:
        cv2.imwrite("{0}erodedots.png".format(imagepath), eroded_ots)
        cv2.imshow('eroded_ots', eroded_ots)
        cv2.waitKey(0)
        cv2.imwrite("{0}dilatedots.png".format(imagepath), dilated_ots)
        cv2.imshow('dilated_ots', dilated_ots)
        cv2.waitKey(0)

    openhull = opening(dilated_ots)

    if show_graphical:
        cv2.imwrite("{0}openhull.png".format(imagepath), openhull)
        cv2.imshow('openhull', openhull)
        cv2.waitKey(0)

    conv_hull = convex_hull_image(openhull)

    ch = np.multiply(conv_hull, 1)  # bool --> binary
    ch = ch.astype('uint8') * 255

    if show_graphical:
        cv2.imwrite("{0}convhull.png".format(imagepath), ch)
        cv2.imshow('conv_hull', ch)
        cv2.waitKey(0)

    bin_ch = (ch / np.max(ch)).astype('uint8')  # binary mask [0, 1]

    if show_graphical:
        cv2.imwrite("{0}mask.png".format(imagepath), ch)
        cv2.imshow('mask', ch)
        cv2.waitKey(0)

    return ch, bin_ch


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


def check_ROI(roi_mask, phantom_image):
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
                        if not re.search('_REPEAT', fname) and not re.search('_PR', fname) and not re.search('_OIL',
                                                                                                             fname):
                            print('Loading ...', fname)

                            # FOR XNAT DOCKER DEVELOPMENT
                            x = re.search('moo', fname)
                            if x:
                                y = 1
                            try:
                                print(y)
                                print('This scan WAS acquired for the SNR test.')
                            except:
                                print('This scan WAS NOT acquired for the SNR test.')
                                exit(1)
                            ########################
                            folder = fname
                            pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                            with os.scandir(pathtodicom) as it:
                                for file in it:
                                    path = "{0}{1}".format(pathtodicom, file.name)

                            ds, imdata, df, dims = dicom_read_and_write(path,
                                                                        writetxt=False)  # function from DICOM_test.py

                            # sd, pn = dicom_geo(ds)

                            try:
                                xdim, ydim = dims
                                print('Matrix Size =', xdim, 'x', ydim)

                                # Sequence parameters required for normalised SNR calculation
                                pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = snr_meta(ds)

                                img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                                cv2.imshow('dicom imdata', img)
                                cv2.waitKey(0)

                            except ValueError:
                                print('DATA INPUT ERROR: this is 3D image data')
                                OrthoSlicer3D(imdata).show()  # look at 3D volume data
                                sys.exit()

    return img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE


def draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, show_graphical=True, imagepath=None):
    """ show_quad = False  # show quadrants for determining signal ROIs on marker image
        show_bbox = False  # show bounding box of phantom on marker image """
    # draw signal ROIs
    # get centre of phantom and definte 5 ROIs from there
    label_img, num = label(bin_mask, connectivity=img.ndim, return_num=True)  # labels the mask

    props = regionprops(label_img)  # returns region properties for phantom mask ROI
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

    # draw bounding box around phantom
    bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
    if show_bbox:
        cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[3], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], bbox[0]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[1], bbox[0]), (255, 255, 255), 1)

    """ DEFINE 4 QUADRANTS (COL,ROW) """
    if show_quad:
        # top left
        cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], pc_row), (0, 0, 255), 1)
        cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 0, 255), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (0, 0, 255), 1)
        cv2.line(marker_im, (pc_col, bbox[0]), (bbox[1], bbox[0]), (0, 0, 255), 1)
        # top right
        cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[3], pc_row), (100, 200, 0), 1)
        cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (100, 200, 0), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (100, 200, 0), 1)
        cv2.line(marker_im, (pc_col, bbox[0]), (bbox[3], bbox[0]), (100, 200, 0), 1)
        # bottom left
        cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[1], pc_row), (0, 140, 255), 1)
        cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 140, 255), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (0, 140, 255), 1)
        cv2.line(marker_im, (pc_col, bbox[2]), (bbox[1], bbox[2]), (0, 140, 255), 1)
        # bottom right
        cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], pc_row), (255, 0, 0), 1)
        cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (255, 0, 0), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (255, 0, 0), 1)
        cv2.line(marker_im, (pc_col, bbox[2]), (bbox[3], bbox[2]), (255, 0, 0), 1)

    """ PUT OTHER 4 ROIs IN CENTRE OF EACH QUADRANT """  # bbox (0min_row, 1min_col, 2max_row, 3max_col)
    # centre coords for each quadrant
    centre1 = [int(((pc_row - bbox[0]) / 2) + bbox[0]), int(((pc_col - bbox[1]) / 2) + bbox[1])]
    centre2 = [int(((pc_row - bbox[0]) / 2) + bbox[0]), int(((bbox[3] - pc_col) / 2) + pc_col)]
    centre3 = [int(((bbox[2] - pc_row) / 2) + pc_row), int(((pc_col - bbox[1]) / 2) + bbox[1])]
    centre4 = [int(((bbox[2] - pc_row) / 2) + pc_row), int(((pc_col - bbox[1]) / 2) + pc_col)]

    quad_centres = [centre1, centre2, centre3, centre4]

    # top left
    cv2.line(marker_im, (centre1[1] + 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] + 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] - 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] + 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] - 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] + 10), (0, 0, 255), 1)
    # top right
    cv2.line(marker_im, (centre2[1] + 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] - 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] + 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] - 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] - 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] + 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] - 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] + 10), (100, 200, 0), 1)
    # bottom left
    cv2.line(marker_im, (centre3[1] + 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] - 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] + 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] - 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] - 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] + 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] - 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] + 10), (0, 140, 255), 1)
    # bottom right
    cv2.line(marker_im, (centre4[1] + 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] - 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] + 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] - 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] - 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] + 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] - 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] + 10), (255, 0, 0), 1)

    if show_graphical:
        cv2.imwrite("{0}drawing_signal_rois.png".format(imagepath), marker_im)
        cv2.imshow('Signal ROIs', marker_im)
        cv2.waitKey(0)

    return pc_row, pc_col, quad_centres, marker_im


def get_signal_value(imdata, pc_row, pc_col, quad_centres):
    # signal values corresponding to voxels inside each signal ROI (don't use greyscale image!)
    signal0 = np.mean(imdata[pc_row - 10:pc_row + 10, pc_col - 10:pc_col + 10])

    centre1 = quad_centres[0]
    centre2 = quad_centres[1]
    centre3 = quad_centres[2]
    centre4 = quad_centres[3]

    signal1 = np.mean(imdata[centre1[0] - 10:centre1[0] + 10, centre1[1] - 10:centre1[1] + 10])
    signal2 = np.mean(imdata[centre2[0] - 10:centre2[0] + 10, centre2[1] - 10:centre2[1] + 10])
    signal3 = np.mean(imdata[centre3[0] - 10:centre3[0] + 10, centre3[1] - 10:centre3[1] + 10])
    signal4 = np.mean(imdata[centre4[0] - 10:centre4[0] + 10, centre4[1] - 10:centre4[1] + 10])

    all_signals = [signal0, signal1, signal2, signal3, signal4]

    mean_signal = np.mean(all_signals)  # mean signal from image data (not filtered!)
    print('Mean signal (total) =', mean_signal)

    return mean_signal, all_signals


def draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=True, imagepath=None,
                         marker_im2=None):
    # Background ROIs according to MagNET protocol
    # TODO: adapt ROI function to correct ROI placement if there is an error re: 20x20 coverage of background ONLY.
    # auto detection of 4 x background ROI samples (one in each corner of background)
    dims = np.shape(mask)
    bin_mask = mask.astype('uint8')

    # for noise stuff
    if marker_im2 == []:
        bin_mask2 = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)  # grayscale to colour
        bin_mask2 = bin_mask2.astype('uint8')

        plt.figure()
        plt.imshow(bin_mask2, cmap='gray')
        plt.axis('off')
        plt.show()

        marker_im2 = marker_im * bin_mask2

    plt.figure()
    plt.imshow(marker_im2, cmap='gray')
    plt.clim(0, 0.01*np.max(marker_im2))
    plt.axis('off')
    plt.show()

    idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
    rows = idx[0]
    cols = idx[1]
    min_row = np.min(rows)  # first row of phantom
    max_row = np.max(rows)  # last row of phantom

    min_col = np.min(cols)  # first column of phantom
    max_col = np.max(cols)  # last column of phantom

    mid_row1 = int(round(min_row / 2))
    mid_row2 = int(round((((dims[0] - max_row) / 2) + max_row)))

    bROI1 = np.zeros(np.shape(mask))  # initialise image matrix for each corner ROI
    bROI2 = np.zeros(np.shape(mask))
    bROI3 = np.zeros(np.shape(mask))
    bROI4 = np.zeros(np.shape(mask))
    bROI5 = np.zeros(np.shape(mask))

    # Background ROIs according to MagNET protocol
    # TODO: adapt ROI function to correct ROI placement if there is an error re: 20x20 coverage of background ONLY.
    if caseT:
        bROI1[mid_row1 - 10:mid_row1 + 10, min_col - 10:min_col + 10] = 255  # top left
        marker_im[mid_row1 - 10:mid_row1 + 10, min_col - 10:min_col + 10] = (0, 0, 255)
        marker_im2[mid_row1 - 10, min_col - 10:min_col + 10] = 255
        marker_im2[mid_row1 + 10, min_col - 10:min_col + 10] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, min_col - 10] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, min_col + 10] = 255
        bROI1_check = check_ROI(bROI1, bin_mask)

        bROI2[mid_row1 - 10:mid_row1 + 10, max_col - 10:max_col + 10] = 255  # top right
        marker_im[mid_row1 - 10:mid_row1 + 10, max_col - 10:max_col + 10] = (0, 255, 0)
        marker_im2[mid_row1 + 10, max_col - 10:max_col + 10] = 255
        marker_im2[mid_row1 - 10, max_col - 10:max_col + 10] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, max_col + 10] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, max_col - 10] = 255
        bROI2_check = check_ROI(bROI2, bin_mask)

        bROI3[mid_row2 - 30:mid_row2 - 10, min_col - 10:min_col + 10] = 255  # bottom left
        marker_im[mid_row2 - 30:mid_row2 - 10, min_col - 10:min_col + 10] = (255, 0, 0)
        marker_im2[mid_row2 - 10, min_col - 10:min_col + 10] = 255
        marker_im2[mid_row2 - 30, min_col - 10:min_col + 10] = 255
        marker_im2[mid_row2 - 30:mid_row2 - 10, min_col + 10] = 255
        marker_im2[mid_row2 - 30:mid_row2 - 10, min_col - 10] = 255
        bROI3_check = check_ROI(bROI3, bin_mask)

        # TODO: check that this fits below phantom.... if not then place on top of phantom
        bROI4[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = 255  # bottom centre
        marker_im[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = (0, 140, 255)
        marker_im2[mid_row2 + 10, pc_col - 10:pc_col + 10] = 255
        marker_im2[mid_row2 - 10, pc_col - 10:pc_col + 10] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, pc_col + 10] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, pc_col - 10] = 255
        bROI4_check = check_ROI(bROI4, bin_mask)

        bROI5[mid_row2 - 30:mid_row2 - 10, max_col - 10:max_col + 10] = 255  # bottom right
        marker_im[mid_row2 - 30:mid_row2 - 10, max_col - 10:max_col + 10] = (205, 235, 255)
        marker_im2[mid_row2 - 10, max_col - 10:max_col + 10] = 255
        marker_im2[mid_row2 - 30, max_col - 10:max_col + 10] = 255
        marker_im2[mid_row2 - 30:mid_row2 - 10, max_col + 10] = 255
        marker_im2[mid_row2 - 30:mid_row2 - 10, max_col - 10] = 255
        bROI5_check = check_ROI(bROI5, bin_mask)

    if caseS or caseC:
        bROI1[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col - 5] = 255  # top left
        marker_im[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col - 5] = (0, 0, 255)
        marker_im2[mid_row1 + 10, min_col - 25:min_col - 5] = 255
        marker_im2[mid_row1 - 10, min_col - 25:min_col - 5] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, min_col - 5] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, min_col - 25] = 255
        bROI1_check = check_ROI(bROI1, bin_mask)

        bROI2[mid_row1 - 10:mid_row1 + 10, max_col + 5:max_col + 25] = 255  # top right
        marker_im[mid_row1 - 10:mid_row1 + 10, max_col + 5:max_col + 25] = (0, 255, 0)
        marker_im2[mid_row1 + 10, max_col + 5:max_col + 25] = 255
        marker_im2[mid_row1 - 10, max_col + 5:max_col + 25] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, max_col + 25] = 255
        marker_im2[mid_row1 - 10:mid_row1 + 10, max_col + 5] = 255
        bROI2_check = check_ROI(bROI2, bin_mask)

        bROI3[mid_row2 - 10:mid_row2 + 10, min_col - 25:min_col - 5] = 255  # bottom left
        marker_im[mid_row2 - 10:mid_row2 + 10, min_col - 25:min_col - 5] = (255, 0, 0)
        marker_im2[mid_row2 + 10, min_col - 25:min_col - 5] = 255
        marker_im2[mid_row2 - 10, min_col - 25:min_col - 5] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, min_col - 5] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, min_col - 25] = 255
        bROI3_check = check_ROI(bROI3, bin_mask)

        # TODO: check that this fits below phantom.... if not then place on top of phantom
        bROI4[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = 255  # bottom centre
        marker_im[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = (0, 140, 255)
        marker_im2[mid_row2 + 10, pc_col - 10:pc_col + 10] = 255
        marker_im2[mid_row2 - 10, pc_col - 10:pc_col + 10] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, pc_col + 10] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, pc_col - 10] = 255
        bROI4_check = check_ROI(bROI4, bin_mask)

        bROI5[mid_row2 - 10:mid_row2 + 10, max_col + 5:max_col + 25] = 255  # bottom right
        marker_im[mid_row2 - 10:mid_row2 + 10, max_col + 5:max_col + 25] = (205, 235, 255)
        marker_im2[mid_row2 + 10, max_col + 5:max_col + 25] = 255
        marker_im2[mid_row2 - 10, max_col + 5:max_col + 25] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, max_col + 25] = 255
        marker_im2[mid_row2 - 10:mid_row2 + 10, max_col + 5] = 255
        bROI5_check = check_ROI(bROI5, bin_mask)

    if show_graphical:
        cv2.imwrite("{0}drawing_bground_rois.png".format(imagepath), marker_im)
        cv2.imshow('Signal and Background ROIs', marker_im)
        cv2.waitKey(0)

    plt.figure()
    plt.imshow(marker_im2, cmap='gray')
    plt.clim(0, 0.01 * np.max(marker_im2))
    plt.axis('off')
    plt.savefig(imagepath + 'noise_and_bROIs.png', bbox_inches='tight')
    plt.show()

    bROIs = [bROI1, bROI2, bROI3, bROI4, bROI5]

    return bROIs


def get_background_noise_value(imdata, bROIs):
    # background/noise voxel values (don't use greyscale image!!)

    bROI1 = bROIs[0]
    bROI2 = bROIs[1]
    bROI3 = bROIs[2]
    bROI4 = bROIs[3]
    bROI5 = bROIs[4]

    n1 = np.std(imdata[np.where(bROI1 == 255)])
    n2 = np.std(imdata[np.where(bROI2 == 255)])
    n3 = np.std(imdata[np.where(bROI3 == 255)])
    n4 = np.std(imdata[np.where(bROI4 == 255)])
    n5 = np.std(imdata[np.where(bROI5 == 255)])

    all_noise = [n1, n2, n3, n4, n5]

    noise = np.mean(all_noise)
    print('Noise in each ROI = ', [n1, n2, n3, n4, n5])
    print('Noise (total) = ', noise)

    return noise, all_noise


def calc_SNR(fact, mean_sig, nse):
    # SNR calculation (background method as opposed to subtraction method)
    SNR_bckgrnd = (fact * mean_sig) / nse
    print('SNR = ', SNR_bckgrnd.round(2))
    return SNR_bckgrnd


def calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor, BW=38.4, BWnom=30):
    # Bandwidth Normalisation
    BWN = np.sqrt(BW) / np.sqrt(BWnom)
    print('Bandwidth normalisation =', BWN.round(2))

    # Voxel Correction - in terms on centimeters to match MagNET Excel report
    # convert from mm to cm
    dx = pixels_space[0] / 10  # ~ 0.09 cm
    dy = pixels_space[1] / 10  # ~ 0.09 cm
    dz = st / 10  # ~ 0.5 cm
    VC = 1 / (dx * dy * dz)
    print('Voxel Correction = ', np.round(VC, 2), 'cm-3')

    # Scan Time Correction - in terms of seconds (not ms)
    STC = 1 / np.sqrt(N_PE * (TR / 1000) * NSA)  # TR in secs
    print('Scan Time Correction = ', STC, 's-1')  # with TR in secs

    # Coil Loading Normalisation
    QN = Qfactor  # depends on test object/coil under investigation
    print('Coil Loading Normalisation = ', QN)

    # Total Correction Factor
    TCF = BWN * VC * STC * QN
    print('Total Correction Factor =', TCF.round(2))

    # Normalised SNR
    NSNR = TCF * SNR_background
    print('Normalised SNR = ', NSNR.round(2))

    return NSNR, BWN, VC, STC, TCF


def draw_spine_signal_ROIs(bin_mask, img, show_bbox=False, show_graphical=True, imagepath=None):
    """ show_bbox = False  # show bounding box of phantom on marker image """
    # draw signal ROIs
    # get centre of phantom and definte 5 ROIs from there
    label_img, num = label(bin_mask, connectivity=img.ndim, return_num=True)  # labels the mask

    props = regionprops(label_img)  # returns region properties for phantom mask ROI
    phantom_centre = props[0].centroid
    pc_row, pc_col = [int(phantom_centre[0]), int(phantom_centre[1])]

    # show detected regions and lines on marker_im
    marker_im = img.copy()
    marker_im = marker_im.astype('uint8')
    marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    cv2.line(marker_im, (pc_col + 18, pc_row + 110), (pc_col + 18, pc_row - 110), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col + 18, pc_row - 110), (pc_col - 17, pc_row - 110), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 17, pc_row - 110), (pc_col - 17, pc_row + 110), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 17, pc_row + 110), (pc_col + 18, pc_row + 110), (0, 0, 255), 1)

    area = ((pc_col + 18) - (pc_col - 17)) * ((pc_row + 110) - (pc_row - 110))
    print('Centre ROI Area =', area)
    area_aim = 220 * 35
    if area != area_aim:
        print('Signal ROI area is too large/too small')
        sys.exit()

    # draw bounding box around phantom
    bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
    if show_bbox:
        cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[3], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], bbox[0]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[1], bbox[0]), (255, 255, 255), 1)

    if show_graphical:
        cv2.imwrite("{0}spine_signal_rois.png".format(imagepath), marker_im)
        cv2.imshow('Signal ROIs', marker_im)
        cv2.waitKey(0)

    return pc_row, pc_col, marker_im


def get_spine_signal_value(imdata, pc_row, pc_col):
    # signal values corresponding to voxels inside each signal ROI (don't use greyscale image!)
    signal0 = np.mean(imdata[pc_row - 110:pc_row + 110, pc_col - 17:pc_col + 18])

    print('Mean signal (total) =', signal0)

    return signal0


def draw_spine_background_ROIs(mask, marker_im, pc_row, show_graphical=True, imagepath=None):
    # Background ROIs according to MagNET protocol
    # auto detection of 4 x background ROI samples (one in each corner of background)
    dims = np.shape(mask)

    idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
    rows = idx[0]
    cols = idx[1]

    min_col = np.min(cols)  # first column of phantom
    max_col = np.max(cols)  # last column of phantom

    row_third = int(round(dims[0] / 3))
    mid_row1 = int(round(row_third / 2))
    mid_row2 = mid_row1 + row_third
    mid_row3 = mid_row2 + row_third

    mid_row4 = int(round(mid_row1 + ((mid_row2 - mid_row1) / 2)))
    mid_row5 = int(round(mid_row2 + ((mid_row3 - mid_row2) / 2)))

    mid_col1 = int(round(min_col / 2))
    mid_col2 = int(round(max_col + ((dims[1] - max_col) / 2)))

    bROI1 = np.zeros(np.shape(mask))  # initialise image matrix for each corner ROI
    bROI2 = np.zeros(np.shape(mask))
    bROI3 = np.zeros(np.shape(mask))
    bROI4 = np.zeros(np.shape(mask))
    bROI5 = np.zeros(np.shape(mask))

    # Background ROIs according to MagNET protocol
    bROI1[pc_row - 5:pc_row + 5, mid_col1 - 5:mid_col1 + 5] = 255  # top left
    marker_im[pc_row - 5:pc_row + 5, mid_col1 - 5:mid_col1 + 5] = (0, 0, 255)

    bROI2[mid_row1 - 5:mid_row1 + 5, mid_col1 - 5:mid_col1 + 5] = 255  # top right
    marker_im[mid_row1 - 5:mid_row1 + 5, mid_col1 - 5:mid_col1 + 5] = (0, 255, 0)

    bROI3[mid_row3 - 5:mid_row3 + 5, mid_col1 - 5:mid_col1 + 5] = 255  # bottom left
    marker_im[mid_row3 - 5:mid_row3 + 5, mid_col1 - 5:mid_col1 + 5] = (255, 0, 0)

    bROI4[mid_row4 - 5:mid_row4 + 5, mid_col2 - 5:mid_col2 + 5] = 255  # bottom centre
    marker_im[mid_row4 - 5:mid_row4 + 5, mid_col2 - 5:mid_col2 + 5] = (0, 140, 255)

    bROI5[mid_row5 - 5:mid_row5 + 5, mid_col2 - 5:mid_col2 + 5] = 255  # bottom right
    marker_im[mid_row5 - 5:mid_row5 + 5, mid_col2 - 5:mid_col2 + 5] = (205, 235, 255)

    if show_graphical:
        cv2.imwrite("{0}spine_all_rois.png".format(imagepath), marker_im)
        cv2.imshow('Signal and Background ROIs', marker_im)
        cv2.waitKey(0)

    bROIs = [bROI1, bROI2, bROI3, bROI4, bROI5]

    return bROIs


def dicom_geo(dicomfile):
    """ extract metadata for scan geometry from Series Description and Protcol Name """

    # Series Description
    series_description = dicomfile[0x0008, 0x103e]
    series_description = series_description.value

    # Protocol Name
    protocol_name = dicomfile[0x0018, 0x1030]
    protocol_name = protocol_name.value

    return series_description, protocol_name
