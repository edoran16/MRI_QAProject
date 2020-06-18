"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import resolution_funcs as rf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import re
import pandas as pd

from scipy.signal import find_peaks, argrelmin
from DICOM_test import dicom_read_and_write
from skimage import filters
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/Resolution_Images/"

show_graphical = False
show_working_plots = False
show_final_plots = True
show_macro_comp = True

matrix_sizes = ['256', '512']
geos = ['_TRA_', '_SAG_', '_COR_']

for ms in range(len(matrix_sizes)):  # iterate through matrix sizes under investigation
    for gs in range(len(geos)):  # iterate through each geometry

        with os.scandir(directpath) as the_folders:
            for folder in the_folders:
                fname = folder.name
                if re.search('-RES_', fname):
                    # iterate between [256, 512] versions of ['COR', 'SAG', 'TRA'] and repeat analysis
                    if re.search(matrix_sizes[ms], fname):
                        if re.search(geos[gs], fname):
                            print('Loading matrix size:', matrix_sizes[ms], 'acquired in', geos[gs], 'geometry...')
                            folder = fname
                            pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                            with os.scandir(pathtodicom) as it:
                                for file in it:
                                    path = "{0}{1}".format(pathtodicom, file.name)

                            ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

        try:
            xdim, ydim = dims
            print('Matrix Size =', xdim, 'x', ydim)

            """ if 256x256 data - only interested in resolving thicker 
            bars, for 512x512 data want to be able to resolve all bars"""
            matrix_dims, pixel_dims = rf.resolution_meta(ds)
            # TODO: use pixel_dims as part of output for report

            img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

            cv2.imshow('dicom imdata', img)
            cv2.waitKey(0)

        except ValueError:
            print('DATA INPUT ERROR: this is 3D image data')
            OrthoSlicer3D(imdata).show()  # look at 3D volume data
            sys.exit()

        # create mask
        mask = rf.create_2D_mask(img)  # watch out for grayscale mask!! [0, 255]
        bin_mask = (mask / np.max(mask)).astype('uint8')  # binary mask [0, 1]

        if show_graphical:
            cv2.imshow('mask', mask)
            cv2.waitKey(0)

        phim = img * bin_mask  # phantom masked image
        bgim = img * (1 - bin_mask)  # background image

        # otsu threshold
        ots = np.zeros_like(phim, dtype=np.uint8)
        ots[(phim > filters.threshold_otsu(phim)) == True] = 255  # Otsu threshold on weighted combination

        if show_graphical:
            cv2.imshow('Otsu Threshold', ots)
            cv2.waitKey(0)

        erode_mask = cv2.erode(bin_mask, None, iterations=2)  # erode the phantom mask to avoid edge effects in TRA/SAG

        sections_only = (255 - ots) * erode_mask

        dilate_sections = cv2.dilate(sections_only, None, iterations=2)
        erode_sections = cv2.erode(dilate_sections, None, iterations=2)  #TODO: check this doesn't need changed back to iterations=2
        label_this = erode_sections

        if show_graphical:
            cv2.imshow('Label This 0', (bin_mask*255).astype('uint8'))
            cv2.waitKey(0)
            cv2.imshow('Label This 1', (erode_mask*255).astype('uint8'))
            cv2.waitKey(0)
            cv2.imshow('Label This 2 ***', sections_only)
            cv2.waitKey(0)
            cv2.imshow('Label This 3', dilate_sections)
            cv2.waitKey(0)
            cv2.imshow('Label This 4', erode_sections)
            cv2.waitKey(0)
            cv2.imshow('Label This 5', label_this)
            cv2.waitKey(0)

        # label this
        label_img, num = label(label_this, connectivity=ots.ndim, return_num=True)  # labels the mask

        if show_graphical:
            plt.figure()
            plt.imshow(label_img)
            plt.show()

        props = regionprops(label_img)  # returns region properties for labelled image
        cent = np.zeros([num, 2])
        areas = np.zeros([num, 1])
        mxAL = np.zeros([num, 1])
        mnAL = np.zeros([num, 1])

        # show detected regions and lines on marker_im
        marker_im = phim.copy()
        marker_im = marker_im.astype('uint8')
        marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

        for xx in range(num):
            cent[xx, :] = props[xx].centroid  # central coordinate
            areas[xx, :] = props[xx].area  # area of detected region
            mxAL[xx, :] = props[xx].major_axis_length
            mnAL[xx, :] = props[xx].minor_axis_length

        cent = cent.astype(int)
        idx = np.where(areas == np.max(areas))
        idx = int(idx[0])

        label_idxs_all = np.arange(1, num + 1)  # list of label numbers - including central section
        label_idxs_where = np.where(label_idxs_all != idx + 1)  # labels to be kept
        label_idxs = label_idxs_all[label_idxs_where]  # identify labels to be kept

        sb = 0  # for subplotting!
        CRFs = []  # contrast response functions
        HV = []  # store horizontal or vertical line label
        bar_minor_axes = []  # store minor axes to determine which sections are 0.5 or 1 mm parallel bars

        # store variables for report output at end of script
        outputs_all = []
        base_signals_all = []
        signal50s_all = []
        min_signals_all = []
        pass_or_fail = []

        for ii in label_idxs:
            block_im = np.zeros(np.shape(label_img))
            block_im[label_img == ii] = 255

            if show_graphical:
                cv2.imshow('Central Block', block_im.astype('uint8'))
                cv2.waitKey(0)

            elements = np.where(block_im == 255)  # [y, x], [rows, cols]

            # top right corner
            min_row = np.min(elements[0])

            # bottom corner
            max_row = np.max(elements[0])

            # top left corner
            min_col = np.min(elements[1])

            # right corner
            max_col = np.max(elements[1])

            # cross section lines
            half_col = int(min_col + ((max_col - min_col) / 2))
            half_row = int(min_row + ((max_row - min_row) / 2))

            # draw the bounding box
            cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
            cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
            cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
            cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

            dist1 = dist.euclidean((half_col, max_row), (half_col, min_row))  # vertical distance
            dist2 = dist.euclidean((min_col, half_row), (max_col, half_row))  # horizontal distance

            case1 = False  # VERTICAL BARS
            case2 = False  # HORIZONTAL BARS

            if dist1 < dist2:  # vertical line is minor axis, bars are horizontal
                case1 = True
                HV.append('H')
                # print('Phase encoding direction.')  # TODO: check this is the case?
            if dist1 > dist2:  # horizontal line is minor axis, bars are vertical
                case2 = True
                HV.append('V')
                # print('Frequency encoding direction.')
            if case1 is False and case2 is False:
                ValueError

            xtrabit = 10  # amount the line is extended over edge to get sense of baseline values

            if case1:  # horizontal parallel bars
                print('HORIZONTAL PARALLEL BARS')
                cv2.line(marker_im, (half_col, max_row + xtrabit), (half_col, min_row - xtrabit), (255, 0, 0),
                         1)  # draw vertical line
                src = (half_col, max_row + xtrabit)
                dst = (half_col, min_row - xtrabit)

                bar_minor_axes.append(mnAL[ii - 1, :])  # minor axes of labelled section

            if case2:  # vertical parallel bars
                print('VERTICAL PARALLEL BARS')
                src = (min_col - xtrabit, half_row)
                dst = (max_col + xtrabit, half_row)
                cv2.line(marker_im, (min_col - xtrabit, half_row), (max_col + xtrabit, half_row), (255, 0, 0),
                         1)  # draw horizontal line

                bar_minor_axes.append(mnAL[ii - 1, :])  # minor axes of labelled section

            cv2.imwrite("{0}marker_image{1}{2}.png".format(imagepath, matrix_sizes[ms], geos[gs]),
                        marker_im.astype('uint8'))
            if show_graphical:
                cv2.imshow('marker image', marker_im.astype('uint8'))
                cv2.waitKey(0)

            """DRAW LINE PROFILE ACROSS PARALLEL BARS"""
            linewidth = 1  # width of the line (mean taken over width)

            # display profile line on phantom: from source code of profile_line function
            src_col, src_row = src = np.asarray(src, dtype=float)
            dst_col, dst_row = dst = np.asarray(dst, dtype=float)
            d_col, d_row = dst - src
            theta = np.arctan2(d_row, d_col)

            length = int(np.ceil(np.hypot(d_row, d_col) + 1))
            """ add one above to include the last point in the profile
            (in contrast to standard numpy indexing) """
            line_col = np.linspace(src_col, dst_col, length)
            line_row = np.linspace(src_row, dst_row, length)

            """ subtract 1 from linewidth to change from pixel-counting
            (make this line 3 pixels wide) to point distances (the
            distance between pixel centers) """
            col_width = (linewidth - 1) * np.sin(-theta) / 2
            row_width = (linewidth - 1) * np.cos(theta) / 2
            perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row])
            perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col])

            improfile = np.copy(img)
            improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 255

            # plot sampled line on phantom to visualise where output comes from
            if show_graphical:
                plt.figure()
                plt.imshow(improfile)
                plt.colorbar()
                plt.axis('off')
                plt.show()

            # voxel values along specified line. specify row then column (y then x)
            output = profile_line(img, (src[1], src[0]), (dst[1], dst[0]))  # signal profile

            baseline_vals = np.append(output[0:xtrabit], output[-(xtrabit + 1):])
            base_signal = np.repeat(np.mean(baseline_vals), len(output))
            min_signal = np.repeat(np.min(output), len(output))
            signal50 = min_signal + ((base_signal - min_signal) / 2)

            # append variables for report output at end of script
            outputs_all.append(output)
            base_signals_all.append(base_signal)
            signal50s_all.append(signal50)
            min_signals_all.append(min_signal)

            analysis_region = output[xtrabit-1:-(xtrabit-1)]  # skip out the two baseline regions
            above50_peaks, _ = find_peaks(analysis_region, height=signal50[0])
            below50_peaks, _ = find_peaks(analysis_region, height=[0, signal50[0]])
            below50_troughs = argrelmin(analysis_region)
            below50_troughs = np.asarray(below50_troughs)
            below50_troughs = below50_troughs[0, :]

            sb = sb + 1

            if show_working_plots:
                plt.figure(sb)
                plt.subplot(1, 2, 1)
                plt.plot(analysis_region)
                plt.plot(above50_peaks, analysis_region[above50_peaks], "x")
                plt.plot(below50_peaks, analysis_region[below50_peaks], "o")
                plt.plot(below50_troughs, analysis_region[below50_troughs], "*")
                plt.plot(np.repeat(signal50[0], len(analysis_region)), "--", color="gray")
                plt.legend(['Line Profile', 'Peaks above 50%', 'Peaks below 50%', 'Troughs below 50%', '50% threshold'],
                        fontsize='xx-small', loc='upper right')

            above50 = np.sum((above50_peaks > 0).astype('uint8'))  # values above the 50% threshold
            below50p = np.sum((below50_peaks > 0).astype('uint8'))  # values below the 50% threshold
            below50t = np.sum((below50_troughs > 0).astype('uint8'))

            # ASSIGNING PASS OR FAIL DEPENDING ON DETECTED PEAKS AND LOCATION OF THOSE PEAKS
            s = 0
            if np.sum(below50t) == 4 and np.sum(above50) >= 3:
                s = 'THIS IS A CLEAR PASS'
            if np.sum(above50) == 0:
                # everything below 50% threshold level
                s = 'THIS IS A CLEAR FAIL'
            if 1 <= np.sum(below50p) < 3 and 1 <= np.sum(above50) < 3:
                # between 1 and 2 peaks below 50%, and between 1 and 2 peaks above 50%
                s = 'THIS IS A BORERLINE PASS'
            if s == 0:
                s = 'OTHER CASE'

            pass_or_fail.append(s)

            # TODO: AMPLITUDE MEASUREMENT IS TO DO WITH DIFFERENCE BETWEEN MEAN AND MAX
            mlp = np.mean(output[xtrabit:-xtrabit])  # mean of profile vals
            alp = np.max(output[xtrabit:-xtrabit]) - np.mean(output[xtrabit:-xtrabit])# amplitude of profile above mlp
            CRF = alp/mlp  # contrast response function
            CRFs.append(CRF)

            if show_working_plots:
                plt.figure(sb)
                plt.subplot(1, 2, 2)
                plt.plot(output)
                plt.plot(base_signal, 'g--')
                plt.plot(signal50, 'k--')
                plt.plot(min_signal, 'r--')
                # TODO: make these right! ESSENTIAL FOR CRF CALCULATION
                # plt.plot(np.repeat(alp, len(output)))
                # plt.plot(np.repeat(mlp, len(output)))
                # plt.text(1, 10, s, fontsize=12)
                plt.title(s)
                plt.legend(['Line Profile', 'Baseline Signal', '50% Signal', 'Minimum Signal', 'alp', 'mlp'],
                        fontsize='xx-small', loc='lower left')

        plt.show()

        """ Identification of which blocks are horizontal/vertical, 0.5 or 1 mm """
        HV_func = lambda HV, hv: [i for (y, i) in zip(hv, range(len(hv))) if HV == y]  # https://pythonspot.com/array-find/
        H_idx = HV_func('H', HV)  # these where lines in horizontal direction
        V_idx = HV_func('V', HV)  # these where lines in vertical direction

        HPBs = []
        VPBs = []
        for ii in range(2):
            HPBs.append(bar_minor_axes[H_idx[ii]])
            VPBs.append(bar_minor_axes[V_idx[ii]])

        thinH = np.where(bar_minor_axes == np.min(HPBs))
        thinH = int(thinH[0])
        thickH = np.where(bar_minor_axes == np.max(HPBs))
        thickH = int(thickH[0])
        thinV = np.where(bar_minor_axes == np.min(VPBs))
        thinV = int(thinV[0])
        thickV = np.where(bar_minor_axes == np.max(VPBs))
        thickV = int(thickV[0])

        """FOR REPORT OUTPUT"""
        if show_final_plots:
            if matrix_dims == [256, 256]:
                plt.figure()
                plt.subplot(121)
                plt.plot(outputs_all[thickH])
                plt.plot(base_signals_all[thickH], 'g--')
                plt.plot(signal50s_all[thickH], 'k--')
                plt.plot(min_signals_all[thickH], 'r--')
                plt.text(0, np.max(outputs_all[thickH]) + 10, pass_or_fail[thickH], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thickH]) + 20])
                plt.title('1 mm Parallel Bars (Horiz.)')
                plt.subplot(122)
                plt.plot(outputs_all[thickV])
                plt.plot(base_signals_all[thickV], 'g--')
                plt.plot(signal50s_all[thickV], 'k--')
                plt.plot(min_signals_all[thickV], 'r--')
                plt.text(0, np.max(outputs_all[thickV]) + 10, pass_or_fail[thickV], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thickV]) + 20])
                plt.title('1 mm Parallel Bars (Vert.)')
                plt.suptitle('256 x 256' + geos[gs] + 'RESULTS')
                plt.show()

            if matrix_dims == [512, 512]:
                plt.figure()
                plt.subplot(121)
                plt.plot(outputs_all[thinH])
                plt.plot(base_signals_all[thinH], 'g--')
                plt.plot(signal50s_all[thinH], 'k--')
                plt.plot(min_signals_all[thinH], 'r--')
                plt.text(0, np.max(outputs_all[thinH]) + 10, pass_or_fail[thinH], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thinH]) + 20])
                plt.title('0.5 mm Parallel Bars (Horiz.)')
                plt.subplot(122)
                plt.plot(outputs_all[thinV])
                plt.plot(base_signals_all[thinV], 'g--')
                plt.plot(signal50s_all[thinV], 'k--')
                plt.plot(min_signals_all[thinV], 'r--')
                plt.text(0, np.max(outputs_all[thinV]) + 10, pass_or_fail[thinV], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thinV]) + 20])
                plt.title('0.5 mm Parallel Bars (Vert.)')
                plt.suptitle('512 x 512' + geos[gs] + 'RESULTS')
                plt.show()

        """CRF PLOT"""  # TODO: work on CRF output. Ideally compare to MTF.
        print('CRF WILL GO HERE')
        # Horizontal CRF plot
        # plt.figure()
        # plt.subplot(121)
        # plt.plot([1, 2], [CRFs[int(thickH[0])], CRFs[int(thinH[0])]], 'o')
        # plt.xlabel('Lines/mm')
        # plt.ylabel('CRF')
        # plt.title('Horizontal Measurement over Parallel Bars')
        # # Vertical CRF plot
        # plt.subplot(122)
        # plt.plot([1, 2], [CRFs[int(thickV[0])], CRFs[int(thinV[0])]], 'o')
        # plt.xlabel('Lines/mm')
        # plt.ylabel('CRF')
        # plt.title('Vertical Measurement over Parallel Bars')
        # plt.show()
        # # print(CRFs)

        # TODO: import results from excel sheet to compare!
        # Comparison with MagNET Report
        if show_macro_comp:
            if matrix_sizes[ms] == '256':
                if geos[gs] == '_TRA_':
                    sheetname = 'Resolution tra 256 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A CLEAR PASS'
                if geos[gs] == '_SAG_':
                    sheetname = 'Resolution_sag_256 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A CLEAR PASS'
                if geos[gs] == '_COR_':
                    sheetname = 'Resolution cor 256 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A CLEAR PASS'
            if matrix_sizes[ms] == '512':
                if geos[gs] == '_TRA_':
                    sheetname = 'Resolution tra 512 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A BORDERLINE PASS'
                if geos[gs] == '_SAG_':
                    sheetname = 'Resolution Sag 512 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A CLEAR PASS'
                if geos[gs] == '_COR_':
                    sheetname = 'Resolution cor 512 Sola'
                    pfh = 'THIS IS A CLEAR PASS'
                    pfv = 'THIS IS A BORDERLINE PASS'

            df = pd.read_excel(r'Sola_INS_07_05_19.xls', sheet_name=sheetname)
            # print(df)
            horiz_data = df.iloc[2:, 0:4]  # + another 3 columns
            horiz_data = horiz_data.dropna()
            vert_data = df.iloc[2:, 13:17]  # + another 3 columns
            vert_data = vert_data.dropna()
            # print(horiz_data)
            # print(vert_data)

            """FOR REPORT OUTPUT"""
            if matrix_dims == [256, 256]:
                print('problem here')
                plt.figure()
                plt.subplot(221)
                plt.plot(outputs_all[thickH])
                plt.plot(base_signals_all[thickH], 'g--')
                plt.plot(signal50s_all[thickH], 'k--')
                plt.plot(min_signals_all[thickH], 'r--')
                plt.text(0, np.max(outputs_all[thickH]) + 10, pass_or_fail[thickH], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thickH]) + 30])
                plt.title('AUTO 1 mm Parallel Bars (Horiz.)')
                plt.subplot(222)
                plt.plot(outputs_all[thickV])
                plt.plot(base_signals_all[thickV], 'g--')
                plt.plot(signal50s_all[thickV], 'k--')
                plt.plot(min_signals_all[thickV], 'r--')
                plt.text(0, np.max(outputs_all[thickV]) + 10, pass_or_fail[thickV], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thickV]) + 30])
                plt.title('AUTO 1 mm Parallel Bars (Vert.)')

                plt.subplot(223)
                plt.plot(horiz_data.iloc[:, 0])
                plt.plot(horiz_data.iloc[:, 1], 'g--')
                plt.plot(horiz_data.iloc[:, 3], 'k--')
                plt.plot(horiz_data.iloc[:, 2], 'r--')
                plt.text(2, np.max(horiz_data.iloc[:, 0]) + 10, pfh, fontsize=12)
                plt.ylim([0, np.max(horiz_data.iloc[:, 0]) + 30])
                plt.title('MACRO 1 mm Parallel Bars (Horiz.)')
                plt.subplot(224)
                plt.plot(vert_data.iloc[:, 0])
                plt.plot(vert_data.iloc[:, 1], 'g--')
                plt.plot(vert_data.iloc[:, 3], 'k--')
                plt.plot(vert_data.iloc[:, 2], 'r--')
                plt.text(2, np.max(vert_data.iloc[:, 0]) + 10, pfv, fontsize=12)
                plt.ylim([0, np.max(vert_data.iloc[:, 0]) + 30])
                plt.title(' MACRO 1 mm Parallel Bars (Vert.)')
                plt.suptitle('256 x 256' + geos[gs] + 'RESULTS')
                plt.show()

            if matrix_dims == [512, 512]:
                plt.figure()
                plt.subplot(221)
                plt.plot(outputs_all[thinH])
                plt.plot(base_signals_all[thinH], 'g--')
                plt.plot(signal50s_all[thinH], 'k--')
                plt.plot(min_signals_all[thinH], 'r--')
                plt.text(0, np.max(outputs_all[thinH]) + 10, pass_or_fail[thinH], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thinH]) + 30])
                plt.title('0.5 mm Parallel Bars (Horiz.)')
                plt.subplot(222)
                plt.plot(outputs_all[thinV])
                plt.plot(base_signals_all[thinV], 'g--')
                plt.plot(signal50s_all[thinV], 'k--')
                plt.plot(min_signals_all[thinV], 'r--')
                plt.text(0, np.max(outputs_all[thinV]) + 10, pass_or_fail[thinV], fontsize=12)
                plt.ylim([0, np.max(outputs_all[thinV]) + 30])
                plt.title('0.5 mm Parallel Bars (Vert.)')

                plt.subplot(223)
                plt.plot(horiz_data.iloc[:, 0])
                plt.plot(horiz_data.iloc[:, 1], 'g--')
                plt.plot(horiz_data.iloc[:, 3], 'k--')
                plt.plot(horiz_data.iloc[:, 2], 'r--')
                plt.text(2, np.max(horiz_data.iloc[:, 0]) + 10, pfh, fontsize=12)
                plt.ylim([0, np.max(horiz_data.iloc[:, 0]) + 30])
                plt.title('MACRO 0.5 mm Parallel Bars (Horiz.)')
                plt.subplot(224)
                plt.plot(vert_data.iloc[:, 0])
                plt.plot(vert_data.iloc[:, 1], 'g--')
                plt.plot(vert_data.iloc[:, 3], 'k--')
                plt.plot(vert_data.iloc[:, 2], 'r--')
                plt.text(2, np.max(vert_data.iloc[:, 0]) + 10, pfv, fontsize=12)
                plt.ylim([0, np.max(vert_data.iloc[:, 0]) + 30])
                plt.title(' MACRO 0.5 mm Parallel Bars (Vert.)')
                plt.suptitle('512 x 512' + geos[gs] + 'RESULTS')
                plt.show()

        """ NEXT ITERATION HERE"""
        # sys.exit()




