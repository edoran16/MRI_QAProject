"""For reference >> https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/"""

import resolution_funcs as rf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
import re
import scipy
from scipy import fft, ifft, interpolate, ndimage

from DICOM_test import dicom_read_and_write
from skimage import filters
from scipy.spatial import distance as dist
from skimage.measure import profile_line, label, regionprops
from nibabel.viewers import OrthoSlicer3D  # << actually do use this!!
from skimage.morphology import opening

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/Resolution_Images/"

show_graphical = False

matrix_sizes = ['256', '512']
geos = ['_TRA_', '_SAG_', '_COR_']

for ms in range(len(matrix_sizes)):  # iterate through matrix sizes under investigation
    for gs in range(len(geos)):  # iterate through each geometry

        with os.scandir(directpath) as the_folders:
            for folder in the_folders:
                fname = folder.name
                if re.search('-RES_', fname):
                    # TODO: iterate between [256, 512] versions of ['COR', 'SAG', 'TRA'] and repeat analysis
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

            # TODO: load dicom metadata here

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

        # cv2.imshow('phantom masked', phim)
        # cv2.waitKey(0)
        # cv2.imshow('background masked', bgim)
        # cv2.waitKey(0)

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
        # print('Number of regions detected = ', num)

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
        # print('Label number to discard = ', idx + 1)
        label_idxs_all = np.arange(1, num + 1)  # list of label numbers - including central section
        label_idxs_where = np.where(label_idxs_all != idx + 1)  # labels to be kept
        label_idxs = label_idxs_all[label_idxs_where]  # identify labels to be kept

        sb = 0  # for subplotting!
        CRFs = []  # contrast response functions
        HV = []  # store horizontal or vertical line label
        bar_minor_axes = []  # store minor axes to determine which sections are 0.5 or 1 mm parallel bars
        for ii in label_idxs:
            block_im = np.zeros(np.shape(label_img))
            block_im[label_img == ii] = 255

            if show_graphical:
                cv2.imshow('Central Block', block_im.astype('uint8'))
                cv2.waitKey(0)
                # print('Major axis length = ', mxAL[ii - 1, :])
                # print('Minor axis length = ', mnAL[ii - 1, :])

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

            # major and minor axes
            dist1 = dist.euclidean((half_col, max_row), (half_col, min_row))  # vertical distance
            dist2 = dist.euclidean((min_col, half_row), (max_col, half_row))  # horizontal distance

            case1 = False
            case2 = False

            if dist1 < dist2:  # vertical line is minor axis
                case1 = True
                HV.append('V')
                # print('Phase encoding direction.')  # TODO: check this is the case?
            if dist1 > dist2:  # horizontal line is minor axis
                case2 = True
                HV.append('H')
                # print('Frequency encoding direction.')
            if case1 is False and case2 is False:
                ValueError

            xtrabit = 10  # amount the line is extended over edge to get sense of baseline values

            if case1:
                # print('CASE1')
                cv2.line(marker_im, (half_col, max_row + xtrabit), (half_col, min_row - xtrabit), (255, 0, 0),
                         1)  # draw vertical line
                src = (half_col, max_row + xtrabit)
                dst = (half_col, min_row - xtrabit)

                bar_minor_axes.append(mnAL[ii - 1, :])  # minor axes of labelled section

            if case2:
                # print('CASE2')
                src = (min_col - xtrabit, half_row)
                dst = (max_col + xtrabit, half_row)
                cv2.line(marker_im, (min_col - xtrabit, half_row), (max_col + xtrabit, half_row), (255, 0, 0),
                         1)  # horizontal line

                bar_minor_axes.append(mnAL[ii - 1, :])  # minor axes of labelled section

            if show_graphical:
                # cv2.imwrite("{0}marker_image.png".format(imagepath), marker_im.astype('uint8'))
                cv2.imshow('marker image', marker_im.astype('uint8'))
                cv2.waitKey(0)

            """DRAW LINE PROFILE ACROSS PARALLEL BARS"""
            # print('Source = ', src)
            # print('Destination = ', dst)

            linewidth = 1  # width of the line (mean taken over width)

            # display profile line on phantom: from source code of profile_line function
            src_col, src_row = src = np.asarray(src, dtype=float)
            dst_col, dst_row = dst = np.asarray(dst, dtype=float)
            d_col, d_row = dst - src
            theta = np.arctan2(d_row, d_col)

            length = int(np.ceil(np.hypot(d_row, d_col) + 1))
            # add one above to include the last point in the profile
            # (in contrast to standard numpy indexing)
            line_col = np.linspace(src_col, dst_col, length)
            line_row = np.linspace(src_row, dst_row, length)

            # subtract 1 from linewidth to change from pixel-counting
            # (make this line 3 pixels wide) to point distances (the
            # distance between pixel centers)
            col_width = (linewidth - 1) * np.sin(-theta) / 2
            row_width = (linewidth - 1) * np.cos(theta) / 2
            perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row])
            perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col])

            improfile = np.copy(img)
            improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 255

            # plot sampled line on phantom to visualise where output comes from
            # plt.figure()
            # plt.imshow(improfile)
            # plt.colorbar()
            # plt.axis('off')
            # plt.show()

            # voxel values along specified line. specify row then column (y then x)
            output = profile_line(img, (src[1], src[0]), (dst[1], dst[0]))  # signal profile

            baseline_vals = np.append(output[0:xtrabit], output[-(xtrabit + 1):])
            base_signal = np.repeat(np.mean(baseline_vals), len(output))
            min_signal = np.repeat(np.min(output), len(output))
            signal50 = min_signal + ((base_signal - min_signal) / 2)

            # TODO: evaluate if pass/fail
            # print(output)
            analysis_region = output[xtrabit-1:-(xtrabit-1)]  # skip out the two baseline regions
            # print(analysis_region)
            import matplotlib.pyplot as plt
            from scipy.signal import find_peaks, argrelmin
            x = analysis_region
            above50_peaks, _ = find_peaks(x, height=signal50[0])
            below50_peaks, _ = find_peaks(x, height=[0, signal50[0]])
            below50_troughs = argrelmin(x)
            below50_troughs = np.asarray(below50_troughs)
            below50_troughs = below50_troughs[0, :]

            sb = sb + 1

            plt.figure(1)
            plt.subplot(2, 2, sb)
            plt.plot(x)
            plt.plot(above50_peaks, x[above50_peaks], "x")
            plt.plot(below50_peaks, x[below50_peaks], "o")
            plt.plot(below50_troughs, x[below50_troughs], "*")
            plt.plot(np.repeat(signal50[0], len(x)), "--", color="gray")
            plt.legend(['Line Profile', 'Peaks above 50%', 'Peaks below 50%', 'Troughs below 50%', '50% threshold'],
                       fontsize='xx-small', loc='upper right')

            above50 = np.sum((above50_peaks > 0).astype('uint8'))  # values above the 50% threshold
            below50p = np.sum((below50_peaks > 0).astype('uint8'))  # values below the 50% threshold
            below50t = np.sum((below50_troughs > 0).astype('uint8'))

            # clear # TODO: change this into case functions
            s = 0
            if np.sum(below50t) == 4 and np.sum(above50) >= 3:
                s = 'THIS IS A CLEAR PASS'
            if np.sum(above50) == 0:  # everything below 50% threshold level
                s = 'THIS IS A CLEAR FAIL'
            if 1 <= np.sum(below50p) < 4 and 1 <= np.sum(above50) < 4:
            # between 1 and 3 peaks below 50%, and between 1 and 3 peaks above 50%
                s = 'THIS IS A BORERLINE PASS'
            if s == 0:
                s = 'OTHER CASE'

            #print(above50, below50, s)

            # if case256:
            # low res, thicker bars resolvable only, 1mm bars only. 7 datapoints. 1 per 1mm voxel
            # thinner bars, 0.5 mm, 3/4 data points, 1 per voxel

            # if case512:
            # high res. thin bars. 0.5mm. 7 datapoints. 1 per 0.5 mm voxel
            # thick bars. 1mm 14 datapoints???
            # TODO: AMPLITUDE MEASUREMENT IS TO DO WITH DIFFERENCE BETWEEN MEAN AND MAX
            mlp = np.mean(output[xtrabit:-xtrabit])  # mean of profile vals
            alp = np.max(output[xtrabit:-xtrabit]) - np.mean(output[xtrabit:-xtrabit])# amplitude of profile above mlp
            # TODO: plot mlp and alp to check the values
            CRF = alp/mlp  # contrast response function
            CRFs.append(CRF)

            plt.figure(2)
            plt.subplot(2, 2, sb)
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

        # TODO: confirm pixel dimensions as well as matrix size from dicom header
        # TODO: identify 4 black line values, determine if abover or below 50% threshold.
        HV_func = lambda HV, hv: [i for (y, i) in zip(hv, range(len(hv))) if HV == y]  # https://pythonspot.com/array-find/
        H_idx = HV_func('H', HV)  # these where lines in horizontal direction
        V_idx = HV_func('V', HV)  # these where lines in vertical direction

        HPBs = []
        VPBs = []
        for ii in range(2):
            HPBs.append(bar_minor_axes[H_idx[ii]])
            VPBs.append(bar_minor_axes[V_idx[ii]])

        thinH = np.where(bar_minor_axes == np.min(HPBs))
        thickH = np.where(bar_minor_axes == np.max(HPBs))
        thinV = np.where(bar_minor_axes == np.min(VPBs))
        thickV = np.where(bar_minor_axes == np.max(VPBs))



        """CRF PLOT"""  # TODO: work on CRF output. Ideally compare to MTF.
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

# TODO: only plot thick bar results for 256 and thin bars for 512
sys.exit()
