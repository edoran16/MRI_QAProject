from DICOM_test import dicom_read_and_write
import cv2
import numpy as np
from scipy import linalg, ndimage
from skimage import filters
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score


def label_img(img):
    """Mask image and label it."""
    val = filters.threshold_otsu(img)  # OTSU threshold to segment phantom
    mask = img > val  # phantom mask
    labels, num = label(mask, connectivity=img.ndim, return_num=True)  # labels the mask

    return labels, num, mask


def get_ref_points(imgs, plotflag=False):
    ref_points = np.zeros([2, 6])
    count = 0
    for im in imgs:
        labels, num, mask = label_img(im)
        props = regionprops(labels)  # returns region properties for phantom mask ROI
        bbox = props[0].bbox
        cen = props[0].centroid
        ref_points[count, :] = np.array([bbox[0], bbox[1], bbox[2], bbox[3], cen[0], cen[1]])
        count = count + 1
        if plotflag:
            cv2.imshow('img', np.float64(mask))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return ref_points


def compute_rigid_transform(ref_points, points):
    """ Computes rotation, scale and translation for aligning
    points to ref_points. """

    p = np.array([[points[0], -points[1], 1, 0],
                  [points[1], points[0], 0, 1],
                  [points[2], -points[3], 1, 0],
                  [points[3], points[2], 0, 1],
                  [points[4], -points[5], 1, 0],
                  [points[5], points[4], 0, 1]])

    y = np.array([ref_points[0],
                  ref_points[1],
                  ref_points[2],
                  ref_points[3],
                  ref_points[4],
                  ref_points[5]])

    # least sq solution to minimize ||Ax-y||
    a, b, tx, ty = linalg.lstsq(p, y)[0]
    r = np.array([[a, -b], [b, a]])  # rotation matrix including scale

    return r, tx, ty


def rigid_alignment(dst, ref_points, pathtosave, plotflag=False):
    """Align images rigidly and save as new images.
    Path determines where the aligned images are saved.
    Set plotFlag=True to plot images."""

    img_points = ref_points[0]
    dst_points = ref_points[1]

    R, TX, TY = compute_rigid_transform(img_points, dst_points)
    T = np.array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

    img2 = ndimage.affine_transform(dst, linalg.inv(T), offset=[-TY, -TX])
    cv2.imwrite(pathtosave + '_phantomcoreg.png', img2*255)

    if plotflag:
        cv2.imshow('NewImg', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # crop away border and save aligned images
    # h, w = img2.shape[:2]
    # border = (w + h)/20

    return img2


def draw_circle_ROI(img, pathtosave):
    # Center coordinates
    center_coordinates = (90, 90)
    # Radius of circle
    radius = 20
    # Color in grayscale
    color = 0
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    image = cv2.circle(img, center_coordinates, radius, color, thickness)
    cv2.imwrite(pathtosave + '_phantom_andROI.png', image)
    # Displaying the image
    cv2.imshow('Phantom + ROI', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # TODO: figure how to return values from ROI. Should be straight forward.
    return image


def isgray(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :,0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False

# TODO: sort out greyscale data Hough transform
def detect_circles(img, pathtosave):
    print(img.dtype, np.min(img), np.max(img))

    # convert img to greyscale
    img2 = img*255
    print(type(img2), img2.dtype, np.min(img2), np.max(img2))
    #img = cv2.medianBlur(img, 5)

    cv2.imshow('Phantom + ROI', img2/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cimg = cv2.cvtColor(np.uint16(img2), cv2.COLOR_GRAY2BGR)
    print(cimg.shape)

    circles = cv2.HoughCircles(np.uint64(img2), cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=30)
    #print(circles)
    #circles = np.uint16(np.around(circles))

    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circles


def replicate_ROI(circles, img):

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


directpath = "data_to_get_started/single_slice_dicom/"  # path to DICOM file
filename = "image1"
path = "{0}{1}".format(directpath, filename)
ds, img, dims = dicom_read_and_write(path)  # function from DICOM_test.py

rows, cols = img.shape
img = img/np.max(img)  # cv2 requires img to be in range 0-1 so normalise image data

# TRANSLATION
M = np.float64([[1, 0, 20], [0, 1, 20]])
dst = cv2.warpAffine(img, M, (cols, rows))

initialplot = True

if initialplot:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(path + '_phantom_img.png', img*255)
    cv2.imwrite(path + '_phantom_dst.png', dst*255)

images = [img, dst]

reference_points = get_ref_points(images, plotflag=True)

img_aligned = rigid_alignment(dst, reference_points, path, plotflag=True)

y_true, skipb, skipa = label_img(img)
y_pred, skipd, skipc = label_img(img_aligned)

j_score = jaccard_score(y_true, y_pred, average='samples')
print('Jaccard similarity score = ', j_score.round(2))

"""DETECTING ROI"""
ROIim = draw_circle_ROI(img, path)
#result = isgray(ROIim)
#print('Is gray = ', result)
#circles = detect_circles(ROIim, path)
#replicate_ROI(circles, img_aligned)




