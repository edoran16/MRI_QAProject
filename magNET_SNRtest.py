""" MagNET SNR measurements. Aim to get code to work on head, body and spine data """

# load SNR TRA NICL
# get code working - then extend to SAG and COR
# then extend to oil phantom
# then look at body coil

import sys
import snr_funcs as sf
import pandas as pd

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/SNR_Images/"

test_object = ['FloodField_HEAD', 'FloodField_BODY']
phantom_type = ['_NICL', '_BODY']
geos = ['_TRA_', '_SAG_', '_COR_']

# TODO: classes will be useful here I think.....

for jj in range(len(test_object)):  # iterate between NICL/flood field and BODY/SPINE
    pt = phantom_type[jj]
    print('SNR analysis of', test_object[jj], 'test object.')

    if jj == 1:  # BODY data
        ii = 0  # only tranverse data to be analsyed
        geometry = geos[ii]
        print('Data geometry =', geometry, '.')

        fullimagepath = imagepath + test_object[jj] + '/' + geometry + '/'
        print(fullimagepath)

        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal

        img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = sf.sort_import_data(directpath, geometry, pt)
        # mask phantom and background
        mask, bin_mask = sf.create_2D_mask(img, show_graphical=True, imagepath=fullimagepath)  # boolean and binary masks
        # draw signal ROIs
        pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False,
                                                                      show_graphical=True, imagepath=fullimagepath)
        # get signal value
        mean_signal, all_signals = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
        factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
        # draw background ROIs
        bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=True, imagepath=fullimagepath)
        # get background/noise value
        b_noise, all_noise = sf.get_background_noise_value(imdata, bROIs)
        # SNR calculation (background method)
        SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)
        # Normalised SNR calculation
        Qfact = 0.84
        NSNR, BWcorr, PixelCorr, TimeCorr, TotalCorr = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background,
                                                                    Qfactor=Qfact)

        # create Pandas data frame with auto results
        auto_data = {'Signal ROI': [1, 2, 3, 4, 5], 'Signal Mean': all_signals,
                     'Background ROI': [1, 2, 3, 4, 5], 'Noise SD': all_noise}

        auto_data2 = {'Mean Signal': mean_signal, 'Mean Noise': b_noise, 'SNR': SNR_background,
                      'Normalised SNR': NSNR}

        auto_df = pd.DataFrame(auto_data, columns=['Signal ROI', 'Signal Mean', 'Background ROI', 'Noise SD'])
        auto_df2 = pd.Series(auto_data2)
        auto_df2.to_frame()

        print(auto_df)
        print(auto_df2)

        auto_constants_data = {'Bandwidth': 38.4, 'Nominal Bandwidth': 30, 'BW Correction': BWcorr,
                               'Pixel Dimensions (mm)': pixels_space, 'Slice width (mm)': st,
                               'Voxel Correction': PixelCorr, 'Phase Encoding Steps': N_PE, 'TR': TR, 'NSA': NSA,
                               'Scan Time Correction': TimeCorr, 'Q Normalisation': Qfact,
                               'Total Correction Factor': TotalCorr}
        auto_constants_df = pd.Series(auto_constants_data)
        auto_constants_df.to_frame()

        print(auto_constants_df)

        # import Excel data with macro results
        excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=1, sheet_name='SNR Body Sola')
        excel_constants_df = excel_df.iloc[1:17, 6:8]
        excel_constants_df = excel_constants_df.dropna(how='all')
        print(excel_constants_df)
        if caseT:
            T_excel_df = excel_df.iloc[0:12, 0:4]
            T_excel_df = T_excel_df.dropna(how='all')
            print(T_excel_df)

    else:  # NICL phantom - analyse all 3 geometries
        for ii in range(len(geos)):
            geometry = geos[ii]
            print('Data geometry =', geometry, '.')

            fullimagepath = imagepath + test_object[jj] + '/' + geometry + '/'
            print(fullimagepath)

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
            mask, bin_mask = sf.create_2D_mask(img, show_graphical=True, imagepath=fullimagepath)  # boolean and binary masks
            # draw signal ROIs
            pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, show_graphical=True, imagepath=fullimagepath)
            # get signal value
            mean_signal, all_signals = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
            factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
            # draw background ROIs
            bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=True, imagepath=fullimagepath)
            # get background/noise value
            b_noise, all_noise = sf.get_background_noise_value(imdata, bROIs)
            # SNR calculation (background method)
            SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)
            # Normalised SNR calculation
            Qfact = 1
            NSNR, BWcorr, PixelCorr, TimeCorr, TotalCorr = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor=Qfact)

            # create Pandas data frame with auto results
            auto_data = {'Signal ROI': [1, 2, 3, 4, 5], 'Signal Mean': all_signals,
                         'Background ROI': [1, 2, 3, 4, 5], 'Noise SD': all_noise}

            auto_data2 = {'Mean Signal': mean_signal, 'Mean Noise': b_noise,  'SNR': SNR_background,
                          'Normalised SNR': NSNR}

            auto_df = pd.DataFrame(auto_data, columns=['Signal ROI', 'Signal Mean', 'Background ROI', 'Noise SD'])
            auto_df2 = pd.Series(auto_data2)
            auto_df2.to_frame()

            print(auto_df)
            print(auto_df2)

            auto_constants_data = {'Bandwidth': 38.4, 'Nominal Bandwidth': 30, 'BW Correction': BWcorr,
                                   'Pixel Dimensions (mm)': pixels_space, 'Slice width (mm)': st,
                                   'Voxel Correction': PixelCorr, 'Phase Encoding Steps': N_PE, 'TR': TR, 'NSA': NSA,
                                   'Scan Time Correction': TimeCorr, 'Q Normalisation': Qfact,
                                   'Total Correction Factor': TotalCorr}
            auto_constants_df = pd.Series(auto_constants_data)
            auto_constants_df.to_frame()

            print(auto_constants_df)

            results_df = auto_df.append(auto_df2, ignore_index=True)
            results_df2 = results_df.append(auto_constants_df, ignore_index=True)
            print('__._ATUTOMATED RESULTS_.__')
            print(results_df2)

            results_df2.to_html('snr_results.html')

            # import Excel data with macro results

            excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=1, sheet_name='SNR Head Head_Ni_Sola')
            print(excel_df)

            excel_constants_df = excel_df.iloc[1:17, 6:8]
            excel_constants_df = excel_constants_df.dropna(how='all')
            # print(excel_constants_df)

            if caseT:
                T_excel_df = excel_df.iloc[0:12, 0:4]
                T_excel_df = T_excel_df.dropna(how='all')
                print(T_excel_df)
            if caseS:
                S_excel_df = excel_df.iloc[14:25, 0:4]
                S_excel_df = S_excel_df.dropna(how='all')
                print(S_excel_df)
            if caseC:
                C_excel_df = excel_df.iloc[28:38, 0:4]
                C_excel_df = C_excel_df.dropna(how='all')
                print(C_excel_df)

            sys.exit()
















