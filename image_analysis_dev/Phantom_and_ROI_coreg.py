from MagNETanalysis.DICOM_test import dicom_read_and_write
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg, ndimage
from skimage import filters
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score

# rigid registration code from O'Reilly Programming Computer Vision with Python textbook. Jan Erik Solem


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
    img3 = img2 + np.abs(np.min(img2))  # rescale
    img4 = img3/np.max(img3)  # normalise
    cv2.imwrite(pathtosave + '_phantomcoreg.png', img2*255)

    if plotflag:
        cv2.imshow('NewImg', img4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img4


def draw_circle_ROI(img, pathtosave, plotflag=False):
    # Center coordinates
    center_coordinates = (90, 90)
    # Radius of circle
    radius = 20
    # Color in grayscale
    color = 255  # ROI detection will not work with color = 0
    # Line thickness of 1 px
    thickness = 1
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    image = cv2.circle(img, center_coordinates, radius, 1, thickness)

    # Displaying the image
    if plotflag:
        cv2.imshow('Phantom + ROI', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(pathtosave + '_phantom_andROI.png', image*255)

    return image


def detect_circles(img_with_ROI, img_orig, pathtosave, plotflag=False):
    """ img = image with ROI on it. ROI must be == 255
        img_orig = original image to draw matched ROI on """

    # convert img to greyscale
    img_gray = img_with_ROI.astype('float32')*255
    img_orig = img_orig.astype('float32')*255

    if plotflag:
        cv2.imshow('Phantom Image with ROI', img_gray.astype('uint8'))
        cv2.waitKey(0)
        cv2.imshow('Image Ready for ROI', img_orig.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    circles = cv2.HoughCircles(np.uint8(img_gray), cv2.HOUGH_GRADIENT, 1, 100, param1=255, param2=10, minRadius=10, maxRadius=50)

    circles = np.uint16(np.round(circles))
    # [coordinate1, coordinate2, radius]
    print(circles)

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img_orig, (i[0], i[1]), i[2], 0, 1)  # 0 = color, 2 = thickness
        # draw the center of the circle
        cv2.circle(img_orig, (i[0], i[1]), 2, 0, 2)

    cv2.imwrite(pathtosave + '_detect_circles.png', img_orig)

    if plotflag:
        cv2.imshow('detected circles', img_orig.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return circles


def replicate_ROI(circle_coords, img_to_draw_on, pathtosave, plotflag=False):

    img_to_draw_ROI_on = img_to_draw_on.astype('float32') * 255

    for i in circle_coords[0, :]:
        # draw the outer circle
        cv2.circle(img_to_draw_ROI_on, (i[0], i[1]), i[2], 1, 1)
        # draw the center of the circle
        cv2.circle(img_to_draw_ROI_on, (i[0], i[1]), 2, 0, 3)

    if plotflag:
        cv2.imshow('ROI Replicated', img_to_draw_ROI_on.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(pathtosave + '_replicateROI.png', img_to_draw_ROI_on)

    return


def get_ROI_voxels(im, roi, plotflag=True):

    print(im.dtype, np.min(im), np.max(im))

    if plotflag:
        cv2.imshow('Draw circle', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    im2 = im.copy()
    im2 = im2*255  # greyscale image of phantom

    roi_mask = np.zeros_like(im)  # create mask for ROI

    for i in roi[0, :]:
        # draw the outer circle
        cv2.circle(im2, (i[0], i[1]), i[2], 255, -1)
        cv2.circle(roi_mask, (i[0], i[1]), i[2], 255, -1)

    if plotflag:
        cv2.imshow('Draw circle', im2.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('ROI Mask', roi_mask.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    mult = im*roi_mask

    if plotflag:
        cv2.imshow('Phantom in ROI', mult.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    voxel_vals = mult[mult > 0]  # voxel values in ROI (in greyscale values!)

    return voxel_vals


directpath = "../data_to_get_started/single_slice_dicom/"  # path to DICOM file
filename = "image1"
path = "{0}{1}".format(directpath, filename)
ds, img, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py
"""img = image from baseline QA with ROIs"""

rows, cols = img.shape
img = img/np.max(img)  # cv2 requires img to be in range 0-1 so normalise image data

# TRANSLATION
M = np.float64([[1, 0, 20], [0, 1, 20]])
dst = cv2.warpAffine(img, M, (cols, rows))
"""dst = routine QA image to be registered with baseline QA scan"""

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

draw_img = img.copy()

ROIim = draw_circle_ROI(draw_img, path, True)
""" ROIim would be the baseline QA image to be matched to."""

circles_detected = detect_circles(ROIim, draw_img, path, True)  # draw_img used here only for plotting

draw_img2 = img_aligned.copy()

replicate_ROI(circles_detected, draw_img2, path, True)

draw_img3 = img_aligned.copy()

ROI_vals = get_ROI_voxels(draw_img3, circles_detected, True)  # greyscale values

# univariate distribution of ROI values
ax = sns.distplot(ROI_vals)
plt.show()




