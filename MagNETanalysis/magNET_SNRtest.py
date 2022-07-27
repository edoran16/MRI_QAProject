""" MagNET SNR measurements. Analysis for head, body coil data data """

from MagNETanalysis import snr_funcs as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directpath = "../MagNET_acceptance_test_data/scans/"
imagepath = "../MagNET_acceptance_test_data/SNR_Images/"
#add a line
test_object = ['FloodField_HEAD', 'FloodField_BODY']
phantom_type = ['_NICL', '_BODY']
geos = ['_TRA_', '_SAG_', '_COR_']

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
        mask, bin_mask = sf.create_2D_mask(img, show_graphical=False, imagepath=fullimagepath)  # boolean and binary masks
        # draw signal ROIs
        pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False,
                                                                      show_graphical=False, imagepath=fullimagepath)
        # get signal value
        mean_signal, all_signals = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
        factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
        # draw background ROIs
        noisemarker = imdata * (1 - bin_mask)
        bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=False, imagepath=fullimagepath, marker_im2=noisemarker)
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

            plt.figure()
            plt.imshow(imdata, cmap='gray')
            plt.clim(0, 0.01 * np.max(imdata))
            plt.axis('off')
            plt.savefig(fullimagepath + 'noise_windowing.png', bbox_inches='tight')
            plt.show()

            # mask phantom and background
            mask, bin_mask = sf.create_2D_mask(img, show_graphical=False, imagepath=fullimagepath)  # boolean and binary masks

            plt.figure()
            plt.imshow(imdata * (1 - bin_mask), cmap='gray')
            plt.clim(0, 0.01*np.max(imdata))
            plt.axis('off')
            plt.savefig(fullimagepath + 'noise_masking.png', bbox_inches='tight')
            plt.show()

            # draw signal ROIs
            pc_row, pc_col, quad_centres, marker_im = sf.draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, show_graphical=False, imagepath=fullimagepath)
            # get signal value
            mean_signal, all_signals = sf.get_signal_value(imdata, pc_row, pc_col, quad_centres)
            factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
            # draw background ROIs
            noisemarker = imdata*(1 - bin_mask)
            bROIs = sf.draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, show_graphical=False, imagepath=fullimagepath, marker_im2=noisemarker)
            # get background/noise value
            b_noise, all_noise = sf.get_background_noise_value(imdata, bROIs)
            # SNR calculation (background method)
            SNR_background = sf.calc_SNR(factor, mean_signal, b_noise)
            # Normalised SNR calculation
            Qfact = 1
            NSNR, BWcorr, PixelCorr, TimeCorr, TotalCorr = sf.calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor=Qfact)

            print('__._AUTOMATED RESULTS_.__')
            # create Pandas data frame with auto results
            auto_data = {'Signal ROI': [1, 2, 3, 4, 5], 'Signal Mean': np.round(all_signals, 2),
                         'Background ROI': [1, 2, 3, 4, 5], 'Noise SD': np.round(all_noise, 2)}

            auto_data2 = {'Mean Signal': np.round(mean_signal, 2), 'Mean Noise': np.round(b_noise, 2),  'SNR': np.round(SNR_background, 2),
                          'Normalised SNR': np.round(NSNR, 2)}

            auto_df = pd.DataFrame(auto_data, columns=['Signal ROI', 'Signal Mean', 'Background ROI', 'Noise SD'])
            auto_df2 = pd.Series(auto_data2)
            auto_df2 = auto_df2.to_frame()

            print(auto_df)
            auto_df.to_html(fullimagepath + 'snr_data.html')
            print(auto_df2)
            auto_df2.to_html(fullimagepath + 'snr_results.html')

            auto_constants_data = {'Bandwidth': 38.4, 'Nominal Bandwidth': 30, 'BW Correction': np.round(BWcorr, 2),
                                   'Pixel Dimensions (mm)': np.round(pixels_space, 2), 'Slice width (mm)': np.round(st, 2),
                                   'Voxel Correction': np.round(PixelCorr, 2), 'Phase Encoding Steps': np.round(N_PE, 2),
                                   'TR': TR, 'NSA': NSA,
                                   'Scan Time Correction': np.round(TimeCorr, 2), 'Q Normalisation': np.round(Qfact, 2),
                                   'Total Correction Factor': np.round(TotalCorr, 2)}
            auto_constants_df = pd.Series(auto_constants_data)
            auto_constants_df = auto_constants_df.to_frame()

            print(auto_constants_df)
            auto_constants_df.to_html(fullimagepath + 'snr_normalisation_constants.html')

            # CONCAT EVERYTHING
            results_df = pd.concat([auto_df, auto_df2], join='outer')
            results_df2 = pd.concat([results_df, auto_constants_df])  # , ignore_index=True
            results_df2 = results_df2.fillna('-')
            print(results_df2)
            results_df2.to_html(fullimagepath + 'snr_results_all.html', justify='center', table_id='TRANVERSE')

            # import Excel data with macro results
            excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=1, sheet_name='SNR Head Head_Ni_Sola')
            print(excel_df)

            excel_constants_df = excel_df.iloc[1:17, 6:8]
            excel_constants_df = excel_constants_df.dropna(how='all')

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

















