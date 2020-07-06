from pylab import *
import cv2
import numpy as np
import os
import snr_funcs as sf
import uni_funcs as uf
from DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/Spine_Images/"

coils = ['_123', '_234', '_345', '_456', '_567', '_678']
all_signals = []
show_graphical = False

for ii in range(len(coils)):
    rx = coils[ii]
    print('Receive coils =', rx, '.')

    with os.scandir(directpath) as the_folders:
        for folder in the_folders:
            fname = folder.name
            if re.search('-SPINE_', fname):
                if re.search(rx, fname):
                    print('Loading ...', fname)
                    folder = fname
                    pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                    with os.scandir(pathtodicom) as it:
                        for file in it:
                            path = "{0}{1}".format(pathtodicom, file.name)

                    ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

                    try:
                        xdim, ydim, zdim = dims
                        # OrthoSlicer3D(imdata).show()  # look at 3D volume data

                        print('Matrix Size =', xdim, 'x', ydim, 'x', zdim)

                        img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                        # Sequence parameters required for normalised SNR calculation
                        pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = sf.snr_meta(ds)

                        sl = int(np.round(xdim/2))  # analysis on slice 8

                        image = img[sl, :, :]

                        if show_graphical:
                            cv2.imshow('dicom imdata', image)
                            cv2.waitKey(0)

                    except ValueError:
                        print('DATA INPUT ERROR: this is 2D image data')
                        sys.exit()

    # mask phantom and background
    mask, bin_mask = sf.create_2D_mask(image, show_graphical=False)  # boolean and binary masks

    # draw signal ROIs
    pc_row, pc_col, marker_im = sf.draw_spine_signal_ROIs(bin_mask, image, show_bbox=True, show_graphical=False)

    # get signal value
    mean_signal = sf.get_spine_signal_value(imdata[sl, :, :], pc_row, pc_col)

    factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
    # draw background ROIs
    bROIs = sf.draw_spine_background_ROIs(mask, marker_im, pc_row, show_graphical=False)

    # get background/noise value
    b_noise = sf.get_background_noise_value(imdata[sl, :, :], bROIs)

    # SNR calculation (background method)
    SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)

    # Normalised SNR calculation
    NSNR = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor=1)

    """ Uniformity Measurement """
    # draw centre ROI
    pc_row, pc_col, marker_im = uf.draw_centre_ROI(bin_mask, image, caseT=False, show_graphical=False)

    # get mean signal value in ROI
    mean_signal = uf.get_signal_value(imdata[sl, :, :], pc_row, pc_col)

    # define uniformity range
    uniformity_range = [mean_signal - (0.1 * mean_signal), mean_signal + (0.1 * mean_signal)]
    print('Expected signal range =', uniformity_range)

    # Obtain Uniformity Profile(s)
    dims = np.shape(image)
    """ plot a vertical profile """
    srcV = (pc_col, 0)  # starting point
    dstV = (pc_col, dims[0] - 1)  # finish point
    all_signalV = uf.obtain_uniformity_profile(imdata[sl, :, :], srcV, dstV, pc_row, pc_col, dist80=0, caseH=False,
                                               caseV=True, show_graphical=False)

    plt.figure()
    plt.plot(all_signalV, 'g')
    plt.plot(np.repeat(uniformity_range[0], len(all_signalV)), 'r--')
    plt.plot(np.repeat(uniformity_range[1], len(all_signalV)), 'r--')
    plt.legend(['Full Profile', 'Lower Range', 'Upper Range'], loc='lower center')
    plt.xlabel('Voxels')
    plt.ylabel('Signal')
    plt.title('Vertical Data')
    plt.show()

    # fractional uniformity calculation
    fractional_uniformity, mean_signal, std_signal = uf.calc_fUniformity(all_signalV, uniformity_range)

    print('Fractional Z Uniformity = ', fractional_uniformity)
    print('Mean Signal =', mean_signal.round(2))
    print('Standard Deviation of Signals =', std_signal.round(2))

    all_signals.append(all_signalV)

plt.figure()
for kk in range(6):
    plt.plot(all_signals[kk])
plt.xlabel('Voxels')
plt.ylabel('Signal')
plt.title('Uniformity Results: FloodField_BODY Test Object')
plt.show()




