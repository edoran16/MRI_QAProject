from pylab import *
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
import cv2
import numpy as np
import os
from DICOM_test import dicom_read_and_write
from nibabel.viewers import OrthoSlicer3D
from scipy.signal import find_peaks, medfilt


def sort_import_data(directpath, geometry):
    with os.scandir(directpath) as the_folders:
        for folder in the_folders:
            fname = folder.name
            if re.search('-GEO_', fname):
                if re.search(geometry, fname):
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

                        img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                        # DICOM INFO
                        matrix_size, st, pixels_space = geo_meta(ds)

                        cv2.imshow('dicom imdata', img)
                        cv2.waitKey(0)

                    except ValueError:
                        print('DATA INPUT ERROR: this is 3D image data')
                        OrthoSlicer3D(imdata).show()  # look at 3D volume data
                        sys.exit()

    return img, imdata, matrix_size, st, pixels_space


def create_2D_mask(img, show_graphical=False, imagepath=None):
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
        cv2.imwrite("{0}mask.png".format(imagepath), ch)
        cv2.imshow('mask', ch)
        cv2.waitKey(0)

    return ch, bin_ch


def geo_meta(dicomfile):
    """ extract metadata for geometric measurements calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # rows and columns
    rows = dicomfile[0x0028, 0x0010]
    rows = rows.value
    cols = dicomfile[0x0028, 0x0011]
    cols = cols.value
    matrix_size = [rows, cols]

    # TODO: FoV and Private Tag Data Access
    # shared_func_groups_seq = dicomfile[0x5200, 0x9229]
    # shared_func_groups_seq = shared_func_groups_seq.value
    # for xx in shared_func_groups_seq:
    #     PrivateTagData = xx[0x002110fe]
    #
    # PrivateTagData = PrivateTagData[0]
    # for yy in PrivateTagData:
    #     #print(yy.value)
    #     print(yy.value)

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # pydicom.dataelem.DataElement
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        st = xx.SliceThickness
        pixels_space = xx.PixelSpacing

    return matrix_size, st, pixels_space


def obtain_profile(imdata, src, dst, caseH, caseV, show_graphical=False, imagepath=None):
    # src and dst are tuples of (x, y) i.e. (column, row)

    # draw line profile across centre line of phantom
    outputs = []
    improfile = np.copy(imdata)
    improfile = (improfile / np.max(improfile))  # normalised
    improfile = improfile * 255  # greyscale
    improfile = improfile.astype('uint8')
    improfile = cv2.cvtColor(improfile, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    # cv2.imshow('test', improfile)

    dims = np.shape(imdata)

    if caseH:  # horizontal lines
        # print('HORIZONTAL PROFILE')  # drawn top of image to bottom of image
        # to get line profile output
        rows = np.repeat(src[1], (dst[0]+1)-src[0])
        cols = np.linspace(src[0], dst[0], (dst[0]+1)-src[0])
    if caseV:  # vertical lines
        # print('VERTICAL PROFILE')  # drawn LHS to RHS of image
        # to get line profile output
        rows = np.linspace(src[1], dst[1], (dst[1]+1)-src[1])
        cols = np.repeat(src[0], (dst[1]+1)-src[1])

        # test = imdata.copy()
        # test[src[1], src[0]] = 15000
        # test[dst[1], dst[0]] = 25000
        # plt.figure()
        # plt.imshow(test)
        # plt.show()

    output = imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)]

    improfile = display_profile_line(improfile, src, dst, caseH, caseV, linecolour=(255, 0, 0), show_graphical=False)

    cv2.imwrite("{0}single_profile.png".format(imagepath), improfile)
    cv2.imshow('Individual Profile Line', improfile)
    cv2.waitKey(0)

    # plot profile line outputs + mean output vs. voxels sampled
    if show_graphical:
        plt.figure()
        plt.plot(output, 'r')
        plt.xlabel('Number of Voxels')
        plt.ylabel('Signal')
        plt.savefig(imagepath + 'signal_profiles_plot.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return output


def display_profile_line(imdata, src, dst, caseH, caseV, linecolour, show_graphical=False, imagepath=None):
    # display profile line on phantom: from source code of profile_line function
    src_col, src_row = np.asarray(src, dtype=float)  # src = (x, y) = (col, row)
    dst_col, dst_row = np.asarray(dst, dtype=float)

    dims = np.shape(imdata)

    if caseH:
        rows = np.repeat(int(src_row), (dst[0]+1)-src[0])
        cols = np.linspace(int(src_col-1), int(dst_col-1), (dst[0]+1)-src[0])

    if caseV:
        rows = np.linspace(int(src_row-1), int(dst_row-1), (dst[1]+1)-src[1])
        cols = np.repeat(int(src_col), (dst[1]+1)-src[1])

    imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)] = linecolour

    # plot sampled line on phantom to visualise where output comes from
    if show_graphical:
        cv2.imwrite("{0}single_line.png".format(imagepath), imdata)
        cv2.imshow('Individual Profile Line!!', imdata)
        cv2.waitKey(0)

    return imdata


def slice_width_calc(profile, pixels_space, st, basefactor=0.25, basefactor2=0.15, show_graphical=False, imagepath=None):
    # normalise profile
    profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))

    if show_graphical:
        plt.plot(profile)
        plt.title('Normalised Profile')
        plt.savefig(imagepath + 'normalised_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # define region of signal profile around plate
    profile_inv = np.max(profile) - profile  # invert profile for peak detection

    if show_graphical:
        plt.plot(profile_inv)
        plt.title('Inverted Profile')
        plt.savefig(imagepath + 'inverted_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    profile_smoothed = medfilt(profile_inv, kernel_size=17)

    if show_graphical:
        plt.plot(profile_smoothed)
        plt.title('Inverted smoothed Profile')
        plt.savefig(imagepath + 'inverted_smoothed_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    profile_inv = profile_smoothed  # increase accuracy for detecting centre of peak

    peaks, _ = find_peaks(profile_inv, height=0.9*(np.max(profile) - np.min(profile)))
    base = int(np.round((basefactor * len(profile_inv))))
    peak_minus = peaks - base
    peak_plus = peaks + base
    profile_cropped = profile[int(peak_minus):int(peak_plus)]

    if show_graphical:
        plt.plot(profile_inv)
        plt.plot(peaks, profile_inv[peaks], "x")
        plt.title('Detecting Extrema to Define Plate and Surrounding Region')
        plt.savefig(imagepath + 'inverted_smoothed_profile_with_peak.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    profile_cropped = profile_cropped / np.max(profile_cropped)

    if show_graphical:
        plt.plot(profile_cropped)
        plt.title('Signal Profile Cropped to Plate')
        plt.savefig(imagepath + 'cropped_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # FWHM
    base2 = int(np.round(basefactor2 * len(profile_cropped)))
    prof_min = np.min(profile_cropped)
    prof_baseline = np.mean(np.append(profile_cropped[:base2], profile_cropped[-base2:]))
    prof_50 = prof_min + ((prof_baseline - prof_min) / 2)

    fwhm_idx = np.where(profile_cropped <= prof_50)
    fwhm_idx_shape = np.shape(fwhm_idx)
    fwhm = fwhm_idx_shape[1]/pixels_space[1]
    print('FWHM = ', fwhm)

    plt.figure(figsize=[10, 7.5])
    plt.plot(profile_cropped, 'r')
    plt.plot(np.min(fwhm_idx), prof_50, 'ko', label='_nolegend_')
    plt.plot(np.max(fwhm_idx), prof_50, 'ko', label='_nolegend_')
    plt.hlines(prof_baseline, 1, len(profile_cropped) - 1, color='g', linestyles='dashdot')
    plt.hlines(prof_min, 1, len(profile_cropped) - 1, color='b', linestyles='dashdot')
    plt.hlines(prof_50, np.min(fwhm_idx), np.max(fwhm_idx), color='k', linestyles='solid')
    plt.vlines(base2, 0, 1, color='c', linestyles='dashed')
    plt.vlines(len(profile_cropped) - base2, 0, 1, color='c', linestyles='dashed')
    plt.text(np.mean(fwhm_idx), prof_50 + 0.05, 'FWHM', fontsize=12, horizontalalignment='center')
    plt.xlabel('Number of Voxels')
    plt.ylabel('Normalised Signal')
    plt.legend(['Plate Profile', 'Baseline', 'Minimum', '50 % Threshold'], loc='center left')
    plt.title('Slice Width Measurement')
    plt.savefig(imagepath + 'slice_width_measurement_figure.png', orientation='landscape', transparent=True,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    stretch_factor = 5
    slice_width = fwhm / stretch_factor

    print('Measured slice width =', slice_width)
    print('Nominal slice width =', st)

    return slice_width

