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
    elem = dicomfile[0x5200, 0x9230]  # pydicom.dataelem.DataElement, (Per-frame Functional Groups Sequence)
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        st = xx.SliceThickness
        slice_space = xx.SpacingBetweenSlices
        pixels_space = xx.PixelSpacing

    return st, slice_space, pixels_space


def create_3D_mask(imdata, dims):
    slice_dim = np.where(dims == np.min(dims))
    slice_dim = slice_dim[0]
    slice_dim = slice_dim[0]
    no_slices = dims[slice_dim]
    print("Number of slices = ", no_slices)
    mask3D = np.zeros_like(imdata)

    for imslice in np.linspace(0, no_slices - 1, no_slices, dtype=int):
        if slice_dim == 0:
            img = imdata[imslice, :, :]  # sagittal
        if slice_dim == 1:
            img = imdata[:, imslice, :]  # coronal
            # TODO: might need to "squeeze out" middle dimension?
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

    return mask3D


def find_centre_and_area_of_phantom(phmask, plotflag=False):
    # centre of phantom
    label_img, num = label(phmask, connectivity=phmask.ndim, return_num=True)  # labels the mask
    props = regionprops(label_img)  # returns region properties for labelled image
    cent = np.zeros([num, 2])
    pharea = np.zeros([num, 1])  # TODO: use this to check that slice is not noise FOV
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
