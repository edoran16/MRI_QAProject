import sys
import uni_funcs as uf
import cv2

from skimage.measure import profile_line, label, regionprops
import matplotlib.pyplot as plt
import numpy as np

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/UNI_Images/"

test_object = ['FloodField', 'Spine']
""" for Flood Field test object - SNR TRA/SAG/COR NICL (OIL data exists as well...)
    for spine test object - SNR TRA BODY """
phantom_type = ['_NICL', '_BODY']
geos = ['_TRA_', '_SAG_', '_COR_']

# TODO: classes will be useful here I think..... change as per SNR test

for jj in range(len(phantom_type)):  # iterate between NICL/flood field and BODY/SPINE
    pt = phantom_type[jj]
    print('Fractional Uniformity analysis of', pt, test_object[jj], 'test object.')

    if jj == 1:  # BODY data
        ii = 0  # only tranverse data to be analsyed
        geometry = geos[ii]
        print('Data geometry =', geometry, '.')
        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal

        img, imdata, pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE = uf.sort_import_data(directpath, geometry, pt)
        # mask phantom and background
        mask, bin_mask = uf.create_2D_mask(img)  # boolean and binary masks
        # draw centre ROI
        pc_row, pc_col, marker_im = uf.draw_centre_ROI(bin_mask, img, caseT, show_graphical=True)

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
                                                   caseV=False, show_graphical=False)
        """ plot vertical profile """
        srcV = (pc_col, 0)  # starting point
        dstV = (pc_col, dims[0] - 1)  # finish point
        all_signalV = uf.obtain_uniformity_profile(imdata, srcV, dstV, pc_row, pc_col, dist80mm, caseH=False,
                                                   caseV=True, show_graphical=False)

        """get 160 mm of horizontal profile"""
        signalH = all_signalH[pc_col - dist80mm:pc_col + dist80mm]

        """get 160 mm of vertical profile """
        signalV = all_signalV[pc_row - dist80mm:pc_row + dist80mm]

        # Check length of signal H and signal V is = 160 mm
        if 161 < (len(signalH) * pixels_space[0]) < 159:
            ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')
        if 161 < (len(signalV) * pixels_space[0]) < 159:
            ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')

        plt.figure()
        plt.subplot(121)
        plt.plot(all_signalH, 'b')
        plt.plot(np.repeat(pc_col, 5), np.linspace(0, np.max(all_signalH), 5), 'y-.')
        plt.plot(np.repeat(pc_col - dist80mm, 5), np.linspace(0, np.max(all_signalH), 5), 'c-.')
        plt.plot(np.repeat(pc_col + dist80mm, 5), np.linspace(0, np.max(all_signalH), 5), 'm-.')
        plt.plot(np.repeat(uniformity_range[0], len(all_signalH)), 'r--')
        plt.plot(np.repeat(uniformity_range[1], len(all_signalH)), 'r--')
        plt.legend(['Full Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Range',
                    'Upper Range'],
                   loc='lower center')
        plt.xlabel('Voxels')
        plt.ylabel('Signal')
        plt.title('Horizontal Data')

        plt.subplot(122)
        plt.plot(all_signalV, 'g')
        plt.plot(np.repeat(pc_row, 5), np.linspace(0, np.max(all_signalV), 5), 'y-.')
        plt.plot(np.repeat(pc_row - dist80mm, 5), np.linspace(0, np.max(all_signalV), 5), 'c-.')
        plt.plot(np.repeat(pc_row + dist80mm, 5), np.linspace(0, np.max(all_signalV), 5), 'm-.')
        plt.plot(np.repeat(uniformity_range[0], len(all_signalV)), 'r--')
        plt.plot(np.repeat(uniformity_range[1], len(all_signalV)), 'r--')
        plt.legend(['Full Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Range',
                    'Upper Range'],
                   loc='lower center')
        plt.xlabel('Voxels')
        plt.ylabel('Signal')
        plt.title('Vertical Data')
        plt.show()

        plt.figure()
        plt.plot(signalH, 'b')
        plt.plot(signalV, 'g')
        plt.plot(np.repeat(uniformity_range[0], len(signalH)), 'r--')
        plt.plot(np.repeat(uniformity_range[1], len(signalH)), 'r--')
        plt.legend(['Horizontal Profile', 'Vertical Profile', 'Expected Lower Range', 'Expected Upper Range'],
                   loc='lower center')
        plt.xlabel('Voxels')
        plt.ylabel('Signal')
        plt.title('Profiles for Fractional Uniformity Calculation')
        plt.show()

        # fractional uniformity calculation
        fractional_uniformityH = uf.calc_fUniformity(signalH, uniformity_range)
        fractional_uniformityV = uf.calc_fUniformity(signalV, uniformity_range)

        if caseT:
            print('Fractional X Uniformity = ', fractional_uniformityH)
            print('Fractional Y Uniformity = ', fractional_uniformityV)

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

            img, imdata, pixels_space = uf.sort_import_data(directpath, geometry, pt)
            # mask phantom and background
            mask, bin_mask = uf.create_2D_mask(img)  # boolean and binary masks
            # draw centre ROI
            pc_row, pc_col, marker_im = uf.draw_centre_ROI(bin_mask, img, caseT, show_graphical=True)

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
            all_signalH = uf.obtain_uniformity_profile(imdata, srcH, dstH, pc_row, pc_col, dist80mm, caseH=True, caseV=False, show_graphical=False)
            """ plot vertical profile """
            srcV = (pc_col, 0)  # starting point
            dstV = (pc_col, dims[0]-1)  # finish point
            all_signalV = uf.obtain_uniformity_profile(imdata, srcV, dstV, pc_row, pc_col, dist80mm, caseH=False, caseV=True, show_graphical=False)

            """get 160 mm of horizontal profile"""
            signalH = all_signalH[pc_col - dist80mm:pc_col + dist80mm]

            """get 160 mm of vertical profile """
            signalV = all_signalV[pc_row - dist80mm:pc_row + dist80mm]

            # Check length of signal H and signal V is = 160 mm
            if 161 < (len(signalH)*pixels_space[0]) < 159:
                ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')
            if 161 < (len(signalV) * pixels_space[0])< 159:
                ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')

            plt.figure()
            plt.subplot(121)
            plt.plot(all_signalH, 'b')
            plt.plot(np.repeat(pc_col, 5), np.linspace(0, np.max(all_signalH), 5), 'y-.')
            plt.plot(np.repeat(pc_col - dist80mm, 5), np.linspace(0, np.max(all_signalH), 5), 'c-.')
            plt.plot(np.repeat(pc_col + dist80mm, 5), np.linspace(0, np.max(all_signalH), 5), 'm-.')
            plt.plot(np.repeat(uniformity_range[0], len(all_signalH)), 'r--')
            plt.plot(np.repeat(uniformity_range[1], len(all_signalH)), 'r--')
            plt.legend(['Full Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Range', 'Upper Range'],
                       loc='lower center')
            plt.xlabel('Voxels')
            plt.ylabel('Signal')
            plt.title('Horizontal Data')

            plt.subplot(122)
            plt.plot(all_signalV, 'g')
            plt.plot(np.repeat(pc_row, 5), np.linspace(0, np.max(all_signalV), 5), 'y-.')
            plt.plot(np.repeat(pc_row - dist80mm, 5), np.linspace(0, np.max(all_signalV), 5), 'c-.')
            plt.plot(np.repeat(pc_row + dist80mm, 5), np.linspace(0, np.max(all_signalV), 5), 'm-.')
            plt.plot(np.repeat(uniformity_range[0], len(all_signalV)), 'r--')
            plt.plot(np.repeat(uniformity_range[1], len(all_signalV)), 'r--')
            plt.legend(['Full Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Range',
                        'Upper Range'],
                       loc='lower center')
            plt.xlabel('Voxels')
            plt.ylabel('Signal')
            plt.title('Vertical Data')
            plt.show()

            plt.figure()
            plt.plot(signalH, 'b')
            plt.plot(signalV, 'g')
            plt.plot(np.repeat(uniformity_range[0], len(signalH)), 'r--')
            plt.plot(np.repeat(uniformity_range[1], len(signalH)), 'r--')
            plt.legend(['Horizontal Profile', 'Vertical Profile', 'Expected Lower Range', 'Expected Upper Range'],
                       loc='lower center')
            plt.xlabel('Voxels')
            plt.ylabel('Signal')
            plt.title('Profiles for Fractional Uniformity Calculation')
            plt.show()

            # fractional uniformity calculation
            fractional_uniformityH, meanH, stdH = uf.calc_fUniformity(signalH, uniformity_range)
            fractional_uniformityV, meanV, stdV = uf.calc_fUniformity(signalV, uniformity_range)

            if caseT:
                print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Y Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')

            if caseS:
                print('Fractional Y Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')

            if caseC:
                print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =', stdH.round(2), ')')
                print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =', stdV.round(2), ')')




