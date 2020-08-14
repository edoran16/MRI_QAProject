import numpy as np
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
from skimage.measure import label, regionprops
import cv2


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def slice_pos_meta(dicomfile):
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # rows and columns
    rows = dicomfile[0x0028, 0x0010]
    rows = rows.value
    cols = dicomfile[0x0028, 0x0011]
    cols = cols.value
    matrix_size = [rows, cols]

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # pydicom.dataelem.DataElement
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        st = xx.SliceThickness
        pixels_space = xx.PixelSpacing

    return matrix_size, st, pixels_space


def create_3D_mask(imdata, dims):
    slice_dim = np.where(dims == np.min(dims))
    slice_dim = slice_dim[0]
    slice_dim = slice_dim[0]
    no_slices = dims[slice_dim]
    mask3D = np.zeros_like(imdata)

    for imslice in np.linspace(0, no_slices - 1, no_slices, dtype=int):
        if slice_dim == 0:
            img = imdata[imslice, :, :]  # sagittal
        if slice_dim == 1:
            img = imdata[:, imslice, :]  # coronal
        if slice_dim == 2:
            img = imdata[:, :, imslice]  # transverse

        h = ex.equalize_hist(img) * 255  # histogram equalisation increases contrast of image
        oi = np.zeros_like(img, dtype=np.uint16)  # creates zero array same dimensions as img
        oi[(img > filters.threshold_otsu(img)) == True] = 1  # Otsu threshold on image
        oh = np.zeros_like(img, dtype=np.uint16)  # zero array same dims as img
        oh[(h > filters.threshold_otsu(h)) == True] = 1  # Otsu threshold on hist eq image

        nm = img.shape[0] * img.shape[1]  # total number of voxels in image
        # calculate normalised weights for weighted combination
        w1 = np.sum(oi) / nm
        w2 = np.sum(oh) / nm
        ots = np.zeros_like(img, dtype=np.uint16)  # create final zero array
        new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
        ots[(new > filters.threshold_otsu(new)) == True] = 1  # Otsu threshold on weighted combination

        openhull = opening(ots)
        conv_hull = convex_hull_image(openhull)
        ch = np.multiply(conv_hull, 1)  # bool --> binary

        if slice_dim == 0:
            mask3D[imslice, :, :] = ch
        if slice_dim == 1:
            mask3D[:, imslice, :] = ch
        if slice_dim == 2:
            mask3D[:, :, imslice] = ch

    return mask3D, no_slices


def find_centre_and_area_of_phantom(phmask, plotflag=False):
    # centre of phantom
    label_img, num = label(phmask, connectivity=phmask.ndim, return_num=True)  # labels the mask
    props = regionprops(label_img)  # returns region properties for labelled image
    cent = np.zeros([num, 2])
    pharea = np.zeros([num, 1])  # used to check that slice is not noise FOV
    for xx in range(num):
        cent[xx, :] = props[xx].centroid  # central coordinate
        pharea[xx, :] = props[xx].area

    cent = np.round(cent).astype(int)
    ph_centre_mark = phmask.copy()
    for ii in cent:
        # draw the center of the circle
        cv2.circle(ph_centre_mark, (ii[1], ii[0]), 1, 0, 1)

    ph_centre_mark = ph_centre_mark * 255

    if plotflag:
        cv2.imshow('centre of the phantom', ph_centre_mark.astype('uint8'))
        cv2.waitKey(0)

    return cent, pharea


def find_range_slice_pos(no_slices, mask3D, imdata, plotflag=False, savepng=False):
    # detect slice range for analysis
    if savepng:
        imagepath = "MagNET_acceptance_test_data/Slice_Position_Images/"

    pf_img_array = []  # array of pass/fail images to be used to create video for example...
    pass_fail = []
    for aa in range(no_slices):
        aa = int(aa)
        phmask = mask3D[aa, :, :]  # phantom mask
        phim = imdata[aa, :, :] * phmask  # phantom image

        ph_centre, pharea = find_centre_and_area_of_phantom(phmask, plotflag=False)

        if pharea > 30000:
            # this slice contains phantom area greater than expected. Likely to be noise. Eliminate for analysis range.
            pass_fail.append(0)

            phim_norm = phim / np.max(phim)  # normalised image
            phim_gray = phim_norm * 255  # greyscale image

            marker_im = phim_gray.copy()
            marker_im = marker_im.astype('uint8')
            marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

            cv2.putText(marker_im, 'FAIL', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pf_img_array.append(marker_im)

            if plotflag:
                cv2.imshow('marker image', marker_im.astype('uint8'))
                cv2.waitKey(0)
            if savepng:
                cv2.imwrite("{0}pass_fail_slice_{1}.png".format(imagepath, aa + 1), marker_im.astype('uint8'))
        else:
            # THICK RECTANGULAR LINE DETECTION
            phim_norm = phim / np.max(phim)  # normalised image
            phim_gray = phim_norm * 255  # greyscale image

            ret, th = cv2.threshold(phim_gray.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th2 = 255 - th
            bigphmask = cv2.erode(phmask.astype('uint8'), None, iterations=3)  # erode phantom mask to elimate edge effect
            bigphmask2 = bigphmask * 255
            th3 = th2 * (phmask * bigphmask2)

            label_img, num = label(th3, connectivity=phim_gray.ndim, return_num=True)  # labels the mask

            props = regionprops(label_img)  # returns region properties for labelled image

            marker_im = phim_gray.copy()
            marker_im = marker_im.astype('uint8')
            marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour
            rcntr = 0  # counter for counting rectangle. Needs to be 1 for slice to PASS
            scntr = 0  # counter for counting 2  squares. If scntr > 0 then slice eliminated
            escntr = 0  # counter for counting edges (inferior/superior) of 2 squares. If escntr > 0 slice eliminated
            for xx in range(num):
                centtemp = props[xx].centroid
                areatemp = props[xx].area
                # conditions for bottom long rectangular shape detection
                if 600 < areatemp < 800:  # TODO: replace with centre of ph coords
                    if centtemp[0] > 170:  # row below (but greater than since -y axis) 170 = bottom region of phantom
                        if 100 < centtemp[1] < 150:  # col in this range is central region of phantom
                            rcntr = rcntr + 1
                            bboxx = props[xx].bbox  # min_row, min_col, max_row, max_col
                            min_row, min_col, max_row, max_col = bboxx
                            # draw the bounding box
                            cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

                # conditions for L and R square shape detection
                if 100 < areatemp < 160:
                    if 110 < centtemp[0] < 150:
                        # rows in central region of phantom
                        if 30 < centtemp[1] < 60 or 190 < centtemp[1] < 220:  # cols in L and R regions of phantom
                            # print('square region detected!')
                            scntr = scntr + 1
                            bboxx = props[xx].bbox  # min_row, min_col, max_row, max_col
                            min_row, min_col, max_row, max_col = bboxx
                            # draw the bounding box
                            cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

                # conditions for edge L and R "square" shape detection
                if 40 < areatemp < 50:
                    if 110 < centtemp[0] < 150:
                        # rows in central region of phantom
                        if 30 < centtemp[1] < 60 or 190 < centtemp[1] < 220:  # cols in L and R regions of phantom
                            escntr = escntr + 1
                            bboxx = props[xx].bbox  # min_row, min_col, max_row, max_col
                            min_row, min_col, max_row, max_col = bboxx
                            # draw the bounding box
                            cv2.line(marker_im, (min_col, min_row), (max_col, min_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, min_row), (max_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (max_col, max_row), (min_col, max_row), (0, 0, 255), 1)
                            cv2.line(marker_im, (min_col, max_row), (min_col, min_row), (0, 0, 255), 1)

            if rcntr == 1 and scntr == 0 and escntr == 0:
                pass_fail.append(1)  # analyse this slice as rectangular region detected only. NO side square regions
                cv2.putText(marker_im, 'PASS', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                pass_fail.append(
                    0)  # eliminate this slice as there is no rectangle OR two side squares have been detected
                cv2.putText(marker_im, 'FAIL', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            pf_img_array.append(marker_im)

            if plotflag:
                cv2.imshow('marker image', marker_im.astype('uint8'))
                cv2.waitKey(0)
            if savepng:
                cv2.imwrite("{0}pass_fail_slice_{1}.png".format(imagepath, aa + 1), marker_im.astype('uint8'))

    passes = np.where(pass_fail)  # passes + 1 = actual slice number
    start_slice = np.min(passes)
    last_slice = np.max(passes)

    return start_slice, last_slice, pf_img_array


def make_video_from_img_array(img_array, dims, VideoName):
    # VideoName must end in .avi
    # dims is a tuple
    # saves avi to current folder
    print('making video ! ... ')
    out = cv2.VideoWriter(VideoName, cv2.VideoWriter_fourcc(*"MP4V"), 2, dims)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return


