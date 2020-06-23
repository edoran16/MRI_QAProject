""" MagNET SNR measurements. Aim to get code to work on head, body and spine data """

# load SNR TRA NICL
# get code working - then extend to SAG and COR
# then extend to oil phantom
# then look at body coil

import sys
import snr_funcs as sf

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/SNR_Images/"

test_object = ['FloodField', 'Spine']
""" for Flood Field test object - SNR TRA/SAG/COR NICL (OIL data exists as well...)
    for spine test object - SNR TRA BODY """
phantom_type = ['_NICL', '_BODY']
geos = ['_TRA_', '_SAG_', '_COR_']

# TODO: classes will be useful here I think.....

for jj in range(len(phantom_type)):  # iterate between NICL/flood field and BODY/SPINE
    pt = phantom_type[jj]
    print('SNR analysis of', pt, test_object[jj], 'test object.')

    if jj == 1:  # BODY data
        ii = 0  # only tranverse data to be analsyed
        geometry = geos[ii]
        print('Data geometry =', geometry, '.')
        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal

        img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = sf.sort_import_data(directpath, geometry, pt)
        # mask phantom and background
        mask, bin_mask = sf.create_2D_mask(img)  # boolean and binary masks
        # draw signal ROIs
        pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False,
                                                                      show_graphical=True)
        # get signal value
        mean_signal = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
        factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
        # draw background ROIs
        bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=True)
        # get background/noise value
        b_noise = sf.get_background_noise_value(imdata, bROIs)
        # SNR calculation (background method)
        SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)
        # Normalised SNR calculation
        NSNR = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor=0.84)

    else:  # NICL phantom - analyse all 3 geometries
        for ii in range(len(geos)):
            geometry = geos[ii]
            print('Data geometry =', geometry, '.')
            if geometry == '_TRA_':
                caseT = True  # transverse
                caseS = False  # sagittal
                caseC = False  # coronal
            if geometry == '_SAG_':
                caseT = False  # transverse
                caseS = True  # sagittal
                caseC = False  # coronal
            if geometry == '_COR_':
                caseT = False  # transverse
                caseS = False  # sagittal
                caseC = True  # coronal

            img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = sf.sort_import_data(directpath, geometry, pt)
            # mask phantom and background
            mask, bin_mask = sf.create_2D_mask(img)  # boolean and binary masks
            # draw signal ROIs
            pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, show_graphical=True)
            # get signal value
            mean_signal = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
            factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
            # draw background ROIs
            bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=True)
            # get background/noise value
            b_noise = sf.get_background_noise_value(imdata, bROIs)
            # SNR calculation (background method)
            SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)
            # Normalised SNR calculation
            NSNR = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor=1)

sys.exit()















