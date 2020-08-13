import geoDL_funcs as gf
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
from scipy.spatial import distance as dist
from skimage import filters
from skimage.measure import label, regionprops

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/GEO_Images/"

geos = ['_TRA', '_SAG', '_COR']

for ii in range(len(geos)):
    show_graphical = True
    geometry = geos[ii]
    print('Data geometry =', geometry, '.')
    if geometry == '_TRA':
        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal
    if geometry == '_SAG':
        caseT = False  # transverse
        caseS = True  # sagittal
        caseC = False  # coronal
    if geometry == '_COR':
        caseT = False  # transverse
        caseS = False  # sagittal
        caseC = True  # coronal

    fullimagepath = imagepath + 'D_and_L/' + geometry[1:] + '/'
    print(fullimagepath)

    img, imdata, matrix_size, st, pixels_space = gf.sort_import_data(directpath, geometry)

    # create mask
    bool_mask, bin_mask = gf.create_2D_mask(img, show_graphical)

    phim = imdata*bin_mask  # phantom image
    bgim = imdata*~bin_mask  # background image

    dims = np.shape(phim)

    phim_norm = phim/np.max(phim)  # normalised image
    phim_gray = (phim_norm*255).astype('uint8')  # greyscale image

    # display phantom image
    cv2.imwrite("{0}geo_phantom_image.png".format(fullimagepath), phim_gray)
    if show_graphical:
        cv2.imshow('phantom image', phim_gray)
        cv2.waitKey(0)

    bigbg = cv2.dilate((255-(bin_mask*255)).astype('uint8'), None, iterations=4)  # dilate background mask

    if show_graphical:
        cv2.imwrite("{0}dilated_background_image.png".format(fullimagepath), bigbg)
        cv2.imshow('Dilated Background', bigbg)
        cv2.waitKey(0)

    # OTSO METHOD
    ots = np.zeros_like(phim_gray, dtype=np.uint16)  # creates zero array same dimensions as img
    ots[(phim_gray > filters.threshold_otsu(phim_gray)) == True] = 1  # Otsu threshold on image

    cv2.imwrite("{0}otsu_thresh_image.png".format(fullimagepath), (ots*255).astype('uint8'))
    cv2.imwrite("{0}inv_otsu_thresh_image.png".format(fullimagepath), ((1 - ots)*255).astype('uint8'))
    cv2.imwrite("{0}inv_otsu_eroded_image.png".format(fullimagepath), (((1 - ots) * ~bigbg)*1).astype('uint8'))

    if show_graphical:
        cv2.imshow('OTS', ots.astype('float32'))
        cv2.waitKey(0)
        cv2.imshow(' INVERSE OTS', (1-ots).astype('float32'))
        cv2.waitKey(0)
        cv2.imshow('INVERSE OTS "ERODED"', ((1 - ots)*~bigbg).astype('float32'))
        cv2.waitKey(0)

    label_this = ((1 - ots)*~bigbg).astype('float32')

    label_img, num = label(label_this, connectivity=phim_gray.ndim, return_num=True)  # labels the mask
    print('Number of regions detected (should be 9!!!) = ', num)

    if num > 9:
        print('Too many regions detected! =O')

    props = regionprops(label_img)  # returns region properties for labelled image
    cent = np.zeros([num, 2])

    marker_im = phim_gray.copy()
    marker_im = marker_im.astype('uint8')
    marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    for xx in range(num):
        cent[xx, :] = props[xx].centroid  # central coordinate

    cent = np.round(cent).astype(int)

    for i in cent:
        # draw the center of the circle
        cv2.circle(marker_im, (i[1], i[0]), 1, (0, 0, 255), 1)

    if show_graphical:
        cv2.imwrite("{0}marker_image.png".format(fullimagepath), marker_im.astype('uint8'))
        cv2.imshow('marker image', marker_im.astype('uint8'))
        cv2.waitKey(0)

    """ START MEASURING HERE """
    toprods = []  # top 3 rods
    midrods = []  # middle 3 rods
    botrods = []  # bottom 3 rods
    centrods = []  # 3 central (vertically) rods

    min_row = np.min(cent[:, 0])
    max_row = np.max(cent[:, 0])
    min_col = np.min(cent[:, 1])
    max_col = np.max(cent[:, 1])

    for i in cent:
        if min_row-10 < i[0] < min_row+10:
            toprods.append(i)
        if max_row-10 < i[0] < max_row+10:
            botrods.append(i)
        if min_row+10 < i[0] < max_row-10:
            midrods.append(i)
        if min_col+10 < i[1] < max_col-10:
            centrods.append(i)

    # source need to be minimum x (left most rod in each pair)
    # dst needs to be maximum x (right most rod in each pair)
    toprods = np.array(toprods)  # horizontal
    midrods = np.array(midrods)  # horizontal
    botrods = np.array(botrods)  # horizontal
    # source need to be minimum y (top most rod in each pair)
    # dst needs to be maximum y (bottom most rod in each pair)
    centrods = np.array(centrods)  # vertical

    # return indexes pf left most and right most markers
    src_top = np.where(toprods[:, 1] == np.min(toprods[:, 1]))  # horizontal
    dst_top = np.where(toprods[:, 1] == np.max(toprods[:, 1]))  # horizontal

    src_mid = np.where(midrods[:, 1] == np.min(midrods[:, 1]))  # horizontal
    dst_mid = np.where(midrods[:, 1] == np.max(midrods[:, 1]))  # horizontal

    src_bot = np.where(botrods[:, 1] == np.min(botrods[:, 1]))  # horizontal
    dst_bot = np.where(botrods[:, 1] == np.max(botrods[:, 1]))  # horizontal

    src_cent = np.where(centrods[:, 0] == np.min(centrods[:, 0]))  # vertical
    dst_cent = np.where(centrods[:, 0] == np.max(centrods[:, 0]))  # vertical

    # index the correct coordinates for markers
    src_t = toprods[src_top]  # horizontal
    dst_t = toprods[dst_top]  # horizontal

    src_m = midrods[src_mid]  # horizontal
    dst_m = midrods[dst_mid]  # horizontal

    src_b = botrods[src_bot]  # horizontal
    dst_b = botrods[dst_bot]  # horizontal

    src_c = centrods[src_cent]  # vertical
    dst_c = centrods[dst_cent]  # vertical

    # make tuple, [x, y] notation replaces [row, col]
    src_t = (src_t[0][1], src_t[0][0])   # horizontal
    dst_t = (dst_t[0][1], dst_t[0][0])  # horizontal

    src_m = (src_m[0][1], src_m[0][0])  # horizontal
    dst_m = (dst_m[0][1], dst_m[0][0])  # horizontal

    src_b = (src_b[0][1], src_b[0][0])  # horizontal
    dst_b = (dst_b[0][1], dst_b[0][0])  # horizontal

    src_c = (src_c[0][1], src_c[0][0])  # vertical
    dst_c = (dst_c[0][1], dst_c[0][0])  # vertical

    # visualise where measurements will be made
    hmarker_im = phim_gray.copy()
    hmarker_im = hmarker_im.astype('uint8')
    hmarker_im = cv2.cvtColor(hmarker_im, cv2.COLOR_GRAY2BGR)  # for horizontal lines

    cv2.line(hmarker_im, src_t, dst_t, (0, 0, 255), 1)  # top line
    cv2.line(hmarker_im, src_m, dst_m, (0, 0, 255), 1)  # middle line
    cv2.line(hmarker_im, src_b, dst_b, (0, 0, 255), 1)  # bottom line

    if show_graphical:
        cv2.imwrite("{0}horiz_marker_image.png".format(fullimagepath), hmarker_im.astype('uint8'))
        cv2.imshow('horiz. marker image', hmarker_im.astype('uint8'))
        cv2.waitKey(0)

    vmarker_im = phim_gray.copy()
    vmarker_im = vmarker_im.astype('uint8')
    vmarker_im = cv2.cvtColor(vmarker_im, cv2.COLOR_GRAY2BGR)  # for vertical lines

    cv2.line(vmarker_im, src_t, src_b, (0, 255, 0), 1)  # left line
    cv2.line(vmarker_im, src_c, dst_c, (0, 255, 0), 1)  # middle line
    cv2.line(vmarker_im, dst_t, dst_b, (0, 255, 0), 1)  # right line

    if show_graphical:
        cv2.imwrite("{0}vert_marker_image.png".format(fullimagepath), vmarker_im.astype('uint8'))
        cv2.imshow('vert. marker image', vmarker_im.astype('uint8'))
        cv2.waitKey(0)

    # compute the Euclidean distance between the exterior markers
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean
    print('Pixel size =', pixels_space[0], 'x', pixels_space[1], 'mm^2')
    # horizontal lines
    hdistt = dist.euclidean(src_t, dst_t)/pixels_space[1]  # top
    hdistm = dist.euclidean(src_m, dst_m)/pixels_space[1]  # middle
    hdistb = dist.euclidean(src_b, dst_b)/pixels_space[1]  # bottom

    cv2.putText(hmarker_im, "{:.1f}mm".format(hdistt), (src_t[0]+4, src_t[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdistm), (src_m[0]+4, src_m[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(hmarker_im, "{:.1f}mm".format(hdistb), (src_b[0]+4, src_b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # vertical lines
    vdistl = dist.euclidean(src_t, src_b)/pixels_space[0]
    vdistc = dist.euclidean(src_c, dst_c)/pixels_space[0]
    vdistr = dist.euclidean(dst_t, dst_b)/pixels_space[0]

    cv2.putText(vmarker_im, "{:.1f}mm".format(vdistl), (src_t[0]+4, src_b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(vmarker_im, "{:.1f}mm".format(vdistc), (src_c[0]+4, dst_c[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(vmarker_im, "{:.1f}mm".format(vdistr), (dst_t[0]+4, dst_b[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    final_markers = cv2.vconcat((hmarker_im.astype('uint8'), vmarker_im.astype('uint8')))

    cv2.imwrite("{0}final_measurements.png".format(fullimagepath), final_markers)
    cv2.imshow("Measurements", final_markers)
    cv2.waitKey(0)

    # error = actual - measured
    errorsH = [120 - hdistt, 120-hdistm, 120 - hdistb]
    errorsH = np.round(errorsH, 2)
    errorsV = [120 - vdistl, 120-vdistc, 120 - vdistr]
    errorsV = np.round(errorsV, 2)

    errors = np.append(errorsH, errorsV)
    pass_fail = []

    for zz in range(len(errors)):
        if np.abs(errors[zz]) > 1:
            print('Error outwith limit for geometric linearity measurement.')
            pass_fail.append('FAIL')
        else:
            pass_fail.append('PASS')

    CV = (np.std(errors)/np.mean(errors))*100
    print('Coefficient of Variation =', CV)

    if -1 >= CV <= 1:
        print('CV within limits for geometric distortion measurement.')

    N = 6
    indH = np.arange(N/2)  # the x locations for the groups
    indV = indH + (N/2)  # the x locations for the groups
    indHV = np.linspace(0, N-1, N)
    ind = np.linspace(-1, N, N+2)
    width = 0.5  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(indH, errorsH, width)
    p2 = plt.bar(indV, errorsV, width)
    p3 = plt.plot(ind, np.repeat(0,  N+2), 'g--')
    p4 = plt.plot(ind, np.repeat(1, N + 2), 'r--')
    p5 = plt.plot(ind, np.repeat(-1, N + 2), 'b--')

    for jj in range(len(errors)):
        plt.text(indHV[jj]-0.25, 0.1, pass_fail[jj], fontsize=12)

    plt.xlabel('Measurement')
    plt.ylabel('Error')
    plt.title('Error on ' + geometry[-3:] + ' H. & V. Measurements')
    plt.xticks(indHV, ('top', 'middle', 'bottom', 'left', 'centre', 'right'))
    plt.legend(['Zero Error', 'Upper Limit', 'Lower Limit', 'Horizontal Errors',
                'Vertical Errors'], ncol=2)
    plt.xlim([-width, N-width])
    plt.ylim([-2, 2])
    plt.savefig(fullimagepath + 'geometric_error.png', orientation='landscape', transparent=True,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # automated results
    # create Pandas data frame with auto results
    auto_data_horiz = {'Line': ['1-3', '4-6', '7-9'],
                 'Distance': [hdistt, hdistm, hdistb],
                 'Error': errorsH,
                 'Result': pass_fail[0:3]}

    auto_df_horiz = pd.DataFrame(auto_data_horiz, columns=['Line', 'Distance', 'Error', 'Result'])
    # vertical metrics
    auto_data_vert = {'Line': ['1-7', '2-8', '3-9'],
                       'Distance': [vdistl, vdistc, vdistr],
                       'Error': errorsV,
                       'Result': pass_fail[-3:]}

    auto_df_vert = pd.DataFrame(auto_data_vert, columns=['Line', 'Distance', 'Error', 'Result'])
    # summary metrics
    auto_distance_summary = {'Geometry': [geometry[-3:], geometry[-3:]],
                             'Direction': ['H', 'V'],
                 'Mean Distance': [np.mean([hdistt, hdistm, hdistb]), np.mean([vdistl, vdistc, vdistr])],
                 'StDev Distance': [np.std([hdistt, hdistm, hdistb]), np.std([vdistl, vdistc, vdistr])],
                 'CoV Distance': [variation([hdistt, hdistm, hdistb])*100, variation([vdistl, vdistc, vdistr])*100]}

    auto_distance_summary_df = pd.DataFrame(auto_distance_summary,
                                            columns=['Geometry', 'Direction', 'Mean Distance',
                                                     'StDev Distance', 'CoV Distance'])

    print('_.__AUTOMATED RESULTS__._')
    print(auto_df_horiz)
    print(auto_df_vert)
    print(auto_distance_summary_df)

    # import Excel data with macro results
    excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=2, sheet_name='Linearity and Distortion Sola')
    excel_df = excel_df.dropna(how='all')  # get rid of NaN rows
    if caseT:
        excel_df = excel_df.iloc[:, 0:4]
    if caseS:
        excel_df = excel_df.iloc[:, 5:9]
    if caseC:
        excel_df = excel_df.iloc[:, 10:14]

    print('_.__MANUAL RESULTS__._')
    print(excel_df)

    """ Slice Width """

    fullimagepath = imagepath + 'SW/' + geometry[1:] + '/'
    print(fullimagepath)

    # plates are located in-between top/middle, middle/bottom rods
    # line profile through plate 1
    src_1_x = src_t[0]
    src_1_y = int(np.round(src_t[1] + ((src_m[1] - src_t[1])/2)))  # [x, y] notation
    dst_1_x = dst_t[0]
    dst_1_y = int(np.round(dst_m[1] + ((dst_t[1] - dst_m[1]) / 2)))  # [x, y] notation

    if caseS:  # sagittal data is rotated pi/2 wrt to transverse and coronal data
        print('SAGITTAL SLICE WIDTH EDIT PLATE 1 COORDINATES')
        src_1_x = int(np.round(src_t[0] + ((src_c[0] - src_t[0]) / 2)))  # [x, y] notation
        src_1_y = src_t[1]
        dst_1_x = src_1_x
        dst_1_y = src_b[1]

    # line profile through plate 2
    src_2_x = src_t[0]
    src_2_y = int(np.round(src_m[1] + ((src_b[1] - src_m[1]) / 2)))  # [x, y] notation
    dst_2_x = dst_t[0]
    dst_2_y = int(np.round(dst_m[1] + ((dst_b[1] - dst_m[1]) / 2)))  # [x, y] notation

    if caseS:  # sagittal data is rotated pi/2 wrt to transverse and coronal data
        print('SAGITTAL SLICE WIDTH EDIT PLATE 2 PROFILE COORDINATES')
        src_2_x = int(np.round(src_c[0] + ((dst_t[0] - src_c[0]) / 2)))  # [x, y] notation
        src_2_y = src_t[1]
        dst_2_x = src_2_x
        dst_2_y = src_b[1]

    swmarker_im = phim_gray.copy()
    swmarker_im = swmarker_im.astype('uint8')
    swmarker_im = cv2.cvtColor(swmarker_im, cv2.COLOR_GRAY2BGR)  # for horizontal lines

    cv2.line(swmarker_im, (src_1_x, src_1_y), (dst_1_x, dst_1_y), (0, 0, 255), 1)  # plate 1 profile
    cv2.line(swmarker_im, (src_2_x, src_2_y), (dst_2_x, dst_2_y), (0, 0, 255), 1)  # plate 2 profile
    show_graphical = False
    if show_graphical:
        cv2.imwrite("{0}slice_width_measurements.png".format(fullimagepath), swmarker_im)
        cv2.imshow("Slice Width Measurements", swmarker_im)
        cv2.waitKey(0)

    # draw line profiles
    if caseS:
        caseH = False
        caseV = True  # slice width profiles are vertical
    else:
        caseH = True  # slice width profiles are horizontal
        caseV = False

    print(caseH, caseV)

    p1_profile = gf.obtain_profile(imdata, (src_1_x, src_1_y), (dst_1_x, dst_1_y), caseH, caseV,
                                   show_graphical=False, imagepath=fullimagepath + 'plate1/')

    p2_profile = gf.obtain_profile(imdata, (src_2_x, src_2_y), (dst_2_x, dst_2_y), caseH, caseV,
                                   show_graphical=False, imagepath=fullimagepath + 'plate2/')

    show_graphical = False

    if show_graphical:
        plt.plot(p1_profile)
        plt.title('Original Plate 1 Profile')
        plt.savefig(fullimagepath + 'plate1/orig_plate1_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.plot(p2_profile)
        plt.title('Original Plate 2 Profile')
        plt.savefig(fullimagepath + 'plate2/orig_plate2_profile.png', orientation='landscape', transparent=True,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    p1_slice_width = gf.slice_width_calc(p1_profile, pixels_space, st, show_graphical=False, imagepath=fullimagepath + 'plate1/')
    p2_slice_width = gf.slice_width_calc(p2_profile, pixels_space, st, show_graphical=False, imagepath=fullimagepath + 'plate2/')

    upperlim = st+(0.1*st)
    lowerlim = st-(0.1*st)

    if upperlim < p1_slice_width < lowerlim:
        print('Slice Width for Plate 1 (', p1_slice_width, ') is OUTWITH tolerance limits.')
        sw_pf1 = 'FAIL'
    else:
        print('Slice Width for Plate 1 (', p1_slice_width, ') is within tolerance limits.')
        sw_pf1 = 'PASS'
    if upperlim < p2_slice_width < lowerlim:
        print('Slice Width for Plate 2 (', p2_slice_width, ') is OUTWITH tolerance limits.')
        sw_pf2 = 'FAIL'
    else:
        print('Slice Width for Plate 2 (', p2_slice_width, ') is within tolerance limits.')
        sw_pf2 = 'PASS'

    # automated results
    # create Pandas data frame with auto results
    auto_data = {'Plate': ['1', '2'], 'Automated Slice Width': [p1_slice_width, p2_slice_width],
                       'Result': [sw_pf1, sw_pf2]}

    auto_df = pd.DataFrame(auto_data, columns=['Plate', 'Automated Slice Width', 'Result'])

    print('_.__AUTOMATED RESULTS__._')
    print(auto_df)

    # import Excel data with macro results
    if caseT:
        excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=2, sheet_name='Slice Width TRA Sola')
        excel_df = excel_df.dropna(how='all')  # get rid of NaN rows
        excel_df1 = excel_df.iloc[0:2, 6:8]
        excel_df2 = excel_df.iloc[0:2, 19:21]
    if caseS:
        excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=2, sheet_name='Slice Width SAG Sola')
        excel_df = excel_df.dropna(how='all')  # get rid of NaN rows
        excel_df1 = excel_df.iloc[0:2, 6:8]
        excel_df2 = excel_df.iloc[0:2, 19:21]
    if caseC:
        excel_df = pd.read_excel(r'Sola_INS_07_05_19.xls', header=2, sheet_name='Slice Width COR Sola')
        excel_df = excel_df.dropna(how='all')  # get rid of NaN rows
        excel_df1 = excel_df.iloc[0:2, 6:8]
        excel_df2 = excel_df.iloc[0:2, 20:22]

    print('_.__MANUAL RESULTS__._')
    print('_.__PLATE 1__._')
    print(excel_df1)
    print('_.__PLATE 2__._')
    print(excel_df2)

