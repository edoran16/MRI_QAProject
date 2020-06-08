import numpy as np
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
from skimage.measure import label, regionprops
import cv2


def create_2D_mask(img):
    """ input:  img is  greyscale uint8 image data from DICOM
        output: ch is 2D mask (also grayscale!!!!)"""

    h = ex.equalize_hist(img)  # histogram equalisation increases contrast of image

    oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
    oi[(img > filters.threshold_otsu(img)) == True] = 255  # Otsu threshold on image

    oh = np.zeros_like(img, dtype=np.uint8)  # zero array same dims as img
    oh[(h > filters.threshold_otsu(h)) == True] = 255  # Otsu threshold on hist eq image

    nm = img.shape[0] * img.shape[1]  # total number of voxels in image
    # calculate normalised weights for weighted combination
    w1 = np.sum(oi) / nm
    w2 = np.sum(oh) / nm
    ots = np.zeros_like(img, dtype=np.uint8)  # create final zero array
    new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
    ots[(new > filters.threshold_otsu(new)) == True] = 255  # Otsu threshold on weighted combination

    openhull = opening(ots)
    conv_hull = convex_hull_image(openhull)

    ch = np.multiply(conv_hull, 1)  # bool --> binary
    ch = ch.astype('uint8')*255

    return ch



