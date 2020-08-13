import uni_funcs as uf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/UNI_Images/"

test_object = ['FloodField_HEAD', 'FloodField_BODY']
""" for Flood Field test object - SNR TRA/SAG/COR NICL (OIL data exists as well...)
    for spine test object - SNR TRA BODY """
phantom_type = ['_NICL', '_BODY']
geos = ['_TRA_', '_SAG_', '_COR_']

# Fractional uniformity/mean uniformity/stdev uniformity vectors for 'append' function
fx = []
fy = []
fz = []
mx = []
my = []
mz = []
sx = []
sy = []
sz = []

print(fx, fy, fz)

for jj in range(len(phantom_type)):
    pt = phantom_type[jj]
    print('Fractional Uniformity analysis of', pt, test_object[jj], 'test object.')

    if jj == 1:  # BODY data
        ii = 0  # only tranverse data to be analsyed
        geometry = geos[ii]
        print('Data geometry =', geometry, '.')
        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal

        fullimagepath = imagepath + test_object[jj] + '/' + geometry + '/'
        print(fullimagepath)

        img, imdata, pixels_space = uf.sort_import_data(directpath, geometry, pt, show_graphical=False, imagepath=fullimagepath)
        # mask phantom and background
        mask, bin_mask = uf.create_2D_mask(img)  # boolean and binary masks
        # draw centre ROI
        pc_row, pc_col, marker_im = uf.draw_centre_ROI(bin_mask, img, caseT, show_graphical=False, imagepath=fullimagepath)

        # get mean signal value in ROI
        mean_signal = uf.get_signal_value(imdata, pc_row, pc_col)

        # define uniformity range
        uniformity_range = [mean_signal - (0.1 * mean_signal), mean_signal + (0.1 * mean_signal)]
        print('Expected signal range =', uniformity_range)

        # Obtain Uniformity Profile(s)
        dims = np.shape(imdata)
        """ define 160 mm region for calculation """
        # +/- 80 mm from centre voxel
        dist80mm = int(np.round(80 / pixels_space[0]))  # how many voxels travserse 80 mm on image
        """plot horizontal profile"""
        srcH = (0, pc_row)  # LHS starting point (x, y) == (col, row)
        dstH = (dims[1] - 1, pc_row)  # RHS finish point
        all_signalH = uf.obtain_uniformity_profile(imdata, srcH, dstH, pc_row, pc_col, dist80mm, caseH=True,
                                                   caseV=False, show_graphical=False, imagepath=fullimagepath)
        """ plot vertical profile """
        srcV = (pc_col, 0)  # starting point
        dstV = (pc_col, dims[0] - 1)  # finish point
        all_signalV = uf.obtain_uniformity_profile(imdata, srcV, dstV, pc_row, pc_col, dist80mm, caseH=False,
                                                   caseV=True, show_graphical=False, imagepath=fullimagepath)

        """get 160 mm of horizontal profile"""
        signalH = all_signalH[pc_col - dist80mm:pc_col + dist80mm]

        """get 160 mm of vertical profile """
        signalV = all_signalV[pc_row - dist80mm:pc_row + dist80mm]

        # Check length of signal H and signal V is = 160 mm
        if 161 < (len(signalH) * pixels_space[0]) < 159:
            ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')
        if 161 < (len(signalV) * pixels_space[0]) < 159:
            ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')

        vlinemaxh = [np.max(all_signalH), uniformity_range[1]]
        vlinemaxh = np.max(vlinemaxh)

        vlinemaxv = [np.max(all_signalV), uniformity_range[1]]
        vlinemaxv = np.max(vlinemaxv)

        plt.figure(figsize=[20, 6])  # width, height in inches
        plt.subplot(121)
        plt.plot(all_signalH, 'b')
        plt.vlines(pc_col, 0, vlinemaxh, colors='y', linestyles='dashdot')
        plt.vlines(pc_col - dist80mm, 0, vlinemaxh, colors='c', linestyles='dashdot')
        plt.vlines(pc_col + dist80mm, 0, vlinemaxh, colors='m', linestyles='dashdot')
        plt.hlines(uniformity_range[0], 0, len(all_signalH), colors='r', linestyles='dashed')
        plt.hlines(uniformity_range[1], 0, len(all_signalH), colors='r', linestyles='dashed')
        plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                    'Upper Limit'], loc='lower left')
        plt.xlabel('Pixel Number')
        plt.ylabel('Signal')
        plt.title('Horizontal Data')

        plt.subplot(122)
        plt.plot(all_signalV, 'g')
        plt.vlines(pc_row, 0, vlinemaxv, colors='y', linestyles='dashdot')
        plt.vlines(pc_row - dist80mm, 0, vlinemaxv, colors='c', linestyles='dashdot')
        plt.vlines(pc_row + dist80mm, 0, vlinemaxv, colors='m', linestyles='dashdot')
        plt.hlines(uniformity_range[0], 0, len(all_signalV), colors='r', linestyles='dashed')
        plt.hlines(uniformity_range[1], 0, len(all_signalV), colors='r', linestyles='dashed')
        plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                    'Upper Limit'], loc='lower left')
        plt.xlabel('Pixel Number')
        plt.ylabel('Signal')
        plt.title('Vertical Data')
        plt.savefig(fullimagepath + 'uniformity_profiles.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

        plt.figure()
        plt.plot(signalH, 'b')
        plt.plot(signalV, 'g')
        plt.hlines(uniformity_range[0], 0, len(signalH), colors='r', linestyles='dashed')
        plt.hlines(uniformity_range[1], 0, len(signalH), colors='r', linestyles='dashed')
        plt.legend(['Horizontal Profile', 'Vertical Profile', 'Lower Limit', 'Upper Limit'],
                   loc='lower left')
        plt.xlabel('Pixel Number')
        plt.ylabel('Signal')
        plt.title('Selected Profile for Fractional Uniformity Calculation')
        plt.savefig(fullimagepath + 'fraction_of_uniformity_profiles.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

        # fractional uniformity calculation
        fractional_uniformityH, meanH, stdH = uf.calc_fUniformity(signalH, uniformity_range)
        fractional_uniformityV, meanV, stdV = uf.calc_fUniformity(signalV, uniformity_range)

        if caseT:
            print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =',
                  stdH.round(2), ')')
            print('Fractional Y Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =',
                  stdV.round(2), ')')
            fx = np.append(fx, fractional_uniformityH)
            fy = np.append(fy, fractional_uniformityV)
            fz = np.append(fz, np.NaN)
            mx = np.append(mx, meanH)
            my = np.append(my, meanV)
            mz = np.append(mz, np.NaN)
            sx = np.append(sx, stdH)
            sy = np.append(sy, stdV)
            sz = np.append(sz, np.NaN)

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

            fullimagepath = imagepath + test_object[jj] + '/' + geometry + '/'
            print(fullimagepath)

            img, imdata, pixels_space = uf.sort_import_data(directpath, geometry, pt, show_graphical=False, imagepath=fullimagepath)
            # mask phantom and background
            mask, bin_mask = uf.create_2D_mask(img)  # boolean and binary masks
            # draw centre ROI
            pc_row, pc_col, marker_im = uf.draw_centre_ROI(bin_mask, img, caseT, show_graphical=False, imagepath=fullimagepath)

            # get mean signal value in ROI
            mean_signal = uf.get_signal_value(imdata, pc_row, pc_col)

            # define uniformity range
            uniformity_range = [mean_signal-(0.1*mean_signal), mean_signal+(0.1*mean_signal)]
            print('Expected signal range =', uniformity_range)

            # Obtain Uniformity Profile(s)
            dims = np.shape(imdata)
            """ define 160 mm region for calculation """
            # +/- 80 mm from centre voxel
            dist80mm = int(np.round(80 / pixels_space[0]))  # how many voxels travserse 80 mm on image
            """plot horizontal profile"""
            srcH = (0, pc_row)  # LHS starting point (x, y) == (col, row)
            dstH = (dims[1]-1, pc_row)  # RHS finish point
            all_signalH = uf.obtain_uniformity_profile(imdata, srcH, dstH, pc_row, pc_col, dist80mm, caseH=True,
                                                       caseV=False, show_graphical=False, imagepath=fullimagepath)
            """ plot vertical profile """
            srcV = (pc_col, 0)  # starting point
            dstV = (pc_col, dims[0]-1)  # finish point
            all_signalV = uf.obtain_uniformity_profile(imdata, srcV, dstV, pc_row, pc_col, dist80mm, caseH=False,
                                                       caseV=True, show_graphical=False, imagepath=fullimagepath)

            """get 160 mm of horizontal profile"""
            signalH = all_signalH[pc_col - dist80mm:pc_col + dist80mm]

            """get 160 mm of vertical profile """
            signalV = all_signalV[pc_row - dist80mm:pc_row + dist80mm]

            # Check length of signal H and signal V is = 160 mm
            if 161 < (len(signalH)*pixels_space[0]) < 159:
                ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')
            if 161 < (len(signalV) * pixels_space[0])< 159:
                ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')

            vlinemaxh = [np.max(all_signalH), uniformity_range[1]]
            vlinemaxh = np.max(vlinemaxh)

            vlinemaxv = [np.max(all_signalV), uniformity_range[1]]
            vlinemaxv = np.max(vlinemaxv)

            plt.figure(figsize=[20, 6])  # width, height in inches
            plt.subplot(121)
            plt.plot(all_signalH, 'b')
            plt.vlines(pc_col, 0, vlinemaxh, colors='y', linestyles='dashdot')
            plt.vlines(pc_col - dist80mm, 0, vlinemaxh, colors='c', linestyles='dashdot')
            plt.vlines(pc_col + dist80mm, 0, vlinemaxh, colors='m', linestyles='dashdot')
            plt.hlines(uniformity_range[0], 0, len(all_signalH), colors='r', linestyles='dashed')
            plt.hlines(uniformity_range[1], 0, len(all_signalH), colors='r', linestyles='dashed')
            plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                        'Upper Limit'], loc='lower left')
            plt.xlabel('Pixel Number')
            plt.ylabel('Signal')
            plt.title('Horizontal Data')

            plt.subplot(122)
            plt.plot(all_signalV, 'g')
            plt.vlines(pc_row, 0, vlinemaxv, colors='y', linestyles='dashdot')
            plt.vlines(pc_row - dist80mm, 0, vlinemaxv, colors='c', linestyles='dashdot')
            plt.vlines(pc_row + dist80mm, 0, vlinemaxv, colors='m', linestyles='dashdot')
            plt.hlines(uniformity_range[0], 0, len(all_signalV), colors='r', linestyles='dashed')
            plt.hlines(uniformity_range[1], 0, len(all_signalV), colors='r', linestyles='dashed')
            plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                        'Upper Limit'], loc='lower left')
            plt.xlabel('Pixel Number')
            plt.ylabel('Signal')
            plt.title('Vertical Data')
            plt.savefig(fullimagepath + 'uniformity_profiles.png', orientation='landscape', transparent=True,
                        bbox_inches='tight', pad_inches=0.1)
            plt.show()

            plt.figure()
            plt.plot(signalH, 'b')
            plt.plot(signalV, 'g')
            plt.hlines(uniformity_range[0], 0, len(signalH), colors='r', linestyles='dashed')
            plt.hlines(uniformity_range[1], 0, len(signalH), colors='r', linestyles='dashed')
            plt.legend(['Horizontal Profile', 'Vertical Profile', 'Lower Limit', 'Upper Limit'],
                       loc='lower left')
            plt.xlabel('Pixel Number')
            plt.ylabel('Signal')
            plt.title('Selected Profile for Fractional Uniformity Calculation')
            plt.savefig(fullimagepath + 'fraction_of_uniformity_profiles.png', orientation='landscape', transparent=True,
                        bbox_inches='tight', pad_inches=0.1)
            plt.show()

            # fractional uniformity calculation
            fractional_uniformityH, meanH, stdH = uf.calc_fUniformity(signalH, uniformity_range)
            fractional_uniformityV, meanV, stdV = uf.calc_fUniformity(signalV, uniformity_range)

            if caseT:
                print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Y Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')
                fx = np.append(fx, fractional_uniformityH)
                fy = np.append(fy, fractional_uniformityV)
                fz = np.append(fz, np.NaN)
                mx = np.append(mx, meanH)
                my = np.append(my, meanV)
                mz = np.append(mz, np.NaN)
                sx = np.append(sx, stdH)
                sy = np.append(sy, stdV)
                sz = np.append(sz, np.NaN)

            if caseS:
                print('Fractional Y Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')
                fx = np.append(fx, np.NaN)
                fy = np.append(fy, fractional_uniformityH)
                fz = np.append(fz, fractional_uniformityV)
                mx = np.append(mx, np.NaN)
                my = np.append(my, meanH)
                mz = np.append(mz, meanV)
                sx = np.append(sx, np.NaN)
                sy = np.append(sy, stdH)
                sz = np.append(sz, stdV)

            if caseC:
                print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')
                fx = np.append(fx, fractional_uniformityH)
                fy = np.append(fy, np.NaN)
                fz = np.append(fz, fractional_uniformityV)
                mx = np.append(mx, meanH)
                my = np.append(my, np.NaN)
                mz = np.append(mz, meanV)
                sx = np.append(sx, stdH)
                sy = np.append(sy, np.NaN)
                sz = np.append(sz, stdV)


# create Pandas data frame with auto results
auto_data = {'Geometry': ['Transverse', 'Sagittal', 'Coronal', 'Transverse'],
             'Coil': ['Head', 'Head', 'Head', 'Body'], 'Signal Range': uniformity_range,
             'Fractional Uniformity X': fx, 'Mean Signal X': mx, 'StDev Signal X': sx,
             'Fractional Uniformity Y': fy, 'Mean Signal Y': my, 'StDev Signal Y': sy,
             'Fractional Uniformity Z': fz, 'Mean Signal Z': mz, 'StDev Signal Z': sz}

auto_df = pd.DataFrame(auto_data, columns=['Geometry', 'Coil', 'Fractional Uniformity X', 'Fractional Uniformity Y',
                                           'Fractional Uniformity Z', 'Mean Signal X', 'Mean Signal Y',
                                           'Mean Signal Z', 'StDev Signal X', 'StDev Signal Y', 'StDev Signal Z'])

print(auto_df.head())

# import Excel data with macro results
excel_head_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=1, sheet_name='Uniformity Head-neck-Ni_Sola')
excel_head_df = excel_head_df.iloc[0:6, 0:15]
excel_head_df = excel_head_df.dropna(how='all')  # get rid of NaN rows
excel_tra_head_df = excel_head_df.iloc[:, 0:2]
excel_sag_head_df = excel_head_df.iloc[:, 7:9]
excel_cor_head_df = excel_head_df.iloc[:, 13:15]

frames = [excel_tra_head_df, excel_sag_head_df, excel_cor_head_df]

manual_df_head_results = pd.concat(frames, ignore_index=True, sort=False, axis=1)
manual_df_head_results.columns = ['(MANUAL) TRA Metric', 'TRA Result',
                                  'SAG Metric', 'SAG Result',
                                  'COR Metric', 'COR Result']
print(manual_df_head_results)

excel_body_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=1, sheet_name='Uniformity Body Sola')
excel_body_df = excel_body_df.iloc[0:6, 0:3]
excel_body_df = excel_body_df.dropna(how='all')
excel_body_df.columns = ['MANUAL TRA', 'BODY', 'RESULTS']
print(excel_body_df)




