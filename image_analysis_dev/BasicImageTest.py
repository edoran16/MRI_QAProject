from MagNETanalysis.DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line, label, regionprops
from skimage import filters, segmentation
from skimage.morphology import binary_erosion, convex_hull_image
from scipy import ndimage
from skimage.draw import ellipse
from skimage import exposure as ex

directpath = "../data_to_get_started/"
folder = "single_slice_dicom/"  # path to DICOM file
filename = "image1"
path = "{0}{1}{2}".format(directpath, folder, filename)
print(path)
ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py
# ds = full DICOM file
# imdata = pixel array
# df = pandas data frame with DICOM header info
# dims = dimensions of pixel array/image

# display image
plt.figure()
plt.imshow(imdata, cmap='bone')
plt.colorbar()
plt.axis('off')
plt.savefig(path + '.png')
plt.show()

# draw line profile across centre line of phantom
src = (dims[0] / 2, 0 + 1)  # starting point
dst = (dims[0] / 2, dims[1] - 1)  # finish point
linewidth = 2  # width of the line (mean taken over width)
output = profile_line(imdata, src, dst)  # voxel values along specified line

# plot profile line output vs. voxel sampled
plt.figure()
plt.plot(output)
plt.xlabel('Voxels')
plt.ylabel('Signal')
plt.savefig(path + '_output.png')
plt.show()

# display profile line on phantom: from source code of profile_line function
src_row, src_col = src = np.asarray(src, dtype=float)
dst_row, dst_col = dst = np.asarray(dst, dtype=float)
d_row, d_col = dst - src
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
perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                  linewidth) for row_i in line_row])
perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                  linewidth) for col_i in line_col])

improfile = np.copy(imdata)
improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 0

# plot sampled line on phantom to visualise where output comes from
plt.figure()
plt.imshow(improfile, cmap='bone')
plt.colorbar()
plt.axis('off')
plt.savefig(path + '_line_profile.png')
plt.show()

# UNIFORMITY MEASUREMENT
# FOREGROUND DETECTION METHOD 1
val = filters.threshold_otsu(imdata)  # OTSU threshold to segment phantom
mask = imdata > val  # phantom mask
# FOREGROUND DETECTION METHOD 2 https://github.com/ccipd/MRQy/blob/master/QCF.py
img = imdata
h = ex.equalize_hist(img[:, :]) * 255  # histogram equalisation increases contrast of image

plt.figure()
plt.subplot(231)
plt.imshow(img, cmap='bone')
plt.title('Original Image')
plt.axis('off')
plt.subplot(232)
plt.imshow(h, cmap='bone')
plt.title('Image after histogram equalisation')
plt.axis('off')

oi = np.zeros_like(img, dtype=np.uint16)  # creates zero array same dimensions as img
oi[(img > filters.threshold_otsu(img)) == True] = 1  # Otsu threshold on image
oh = np.zeros_like(img, dtype=np.uint16)  # zero array same dims as img
oh[(h > filters.threshold_otsu(h)) == True] = 1  # Otsu threshold on hist eq image

plt.subplot(233)
plt.imshow(oi, cmap='bone')
plt.title('Image Otsu Threshold')
plt.axis('off')
plt.subplot(234)
plt.imshow(oh, cmap='bone')
plt.title('Hist Eq Image Threshold')
plt.axis('off')

nm = img.shape[0] * img.shape[1]  # total number of voxels in image
# calculate normalised weights for weighted combination
w1 = np.sum(oi) / nm
w2 = np.sum(oh) / nm
ots = np.zeros_like(img, dtype=np.uint16)  # create final zero array
new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
ots[(new > filters.threshold_otsu(new)) == True] = 1  # Otsu threshold on weighted combination

plt.subplot(235)
plt.imshow(ots, cmap='bone')
plt.title('Weighted Combo Image Threshold')
plt.axis('off')

# not sure this stage is that helpful... doesn't mask top of phantom right
conv_hull = convex_hull_image(ots)  # set of pixels included in smallest convex polygon that SURROUND all
# white pixels in the input image
ch = np.multiply(conv_hull, 1)  # bool --> binary

plt.subplot(236)
plt.imshow(ch, cmap='bone')
plt.title('Convex Hull Image')
plt.axis('off')
plt.show()

ch = ots  # ignore convex hull step
fore_image = ch * img  # phantom
back_image = (1 - ch) * img  # background

plt.figure()
plt.subplot(221)
plt.imshow(mask)
plt.title('Otsu Mask')
plt.axis('off')
plt.subplot(222)
plt.imshow(imdata * ~mask)
plt.title('Otsu Background')
plt.axis('off')
plt.subplot(223)
plt.imshow(fore_image)
plt.title('Foreground Image')
plt.axis('off')
plt.subplot(224)
plt.imshow(back_image)
plt.axis('off')
plt.title('Background Image')
plt.tight_layout()
plt.savefig(path + '_foreground_detection2.png')
plt.show()
###########################

phantom_edges = segmentation.find_boundaries(mask, mode='thin').astype(np.uint8)  # finds outline of mask

plt.figure()
plt.subplot(131)
plt.imshow(mask)
plt.title('Otsu Mask')
plt.axis('off')
plt.subplot(132)
plt.imshow(phantom_edges)
plt.title('Phantom Boundary')
plt.axis('off')
plt.subplot(133)
plt.imshow(mask + phantom_edges)
plt.axis('off')
plt.title('Overlay')
plt.tight_layout()
plt.savefig(path + '_otsu_boundary_mask.png')
plt.show()

label_img, num = label(mask, connectivity=imdata.ndim, return_num=True)  # labels the mask
props = regionprops(label_img)  # returns region properties for phantom mask ROI
area100 = props[0].area  # area of phantom mask
area80 = area100 * 0.8  # desired phantom area = 80% of phantom area [IPEM Report 112]

ROIerode = []  # initialise variable for eroded ROI (80% ROI)
temp_mask = np.copy(mask)  # copy of phantom mask
old_mask_area = np.sum(temp_mask)  # 100% area

new_mask_area = old_mask_area  # initialise new_mask_area to be updated in while loop
count = 0  # initialise counter for while loop
while new_mask_area > area80:  # whilst the area is greater than 80% of original area continue eroding mask
    count = count + 1  # counter
    shrunk_mask = binary_erosion(temp_mask)
    unraveled_mask = np.ravel(shrunk_mask)
    new_mask_area = np.sum(unraveled_mask)
    temp_mask = np.reshape(unraveled_mask, dims)

print('No. of iterations = ', count)  # number of iterations to reduce mask to desired 80% area
print('Updated area is ', round((new_mask_area / old_mask_area) * 100, 2), '% of original phantom mask')
ROIerode = temp_mask  # eroded mask from while loop

plt.figure()
plt.subplot(121)
plt.imshow(mask)
plt.axis('off')
plt.subplot(122)
plt.imshow(mask * ~ROIerode)
plt.axis('off')
plt.savefig(path + '_eroded_mask.png')
plt.show()

# LOW PASS FILTER APPLIED TO REDUCE EFFECTS OF NOISE
# not crucial if SNR is high
# specified as pre-processing step in IPEM Report 112

a = imdata.copy()  # copy of DICOM image
k = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # kernel
imdata_conv = ndimage.convolve(a, k, mode='constant', cval=0.0)  # convolution of image and kernel

# display image and filtered image (normalised)
plt.figure()
plt.subplot(121)
plt.imshow(imdata / np.max(imdata), cmap='bone')
plt.title('Original Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(imdata_conv / np.max(imdata_conv), cmap='bone')
plt.title('Low Pass Filtered Image')
plt.axis('off')
plt.savefig(path + '_convolution.png')
plt.show()

mask_uniformity = ROIerode  # mask for measuring uniformity is the eroded mask
RoiVoxelVals1 = []  # initialise variable for voxel vals from filtered image data
RoiVoxelVals2 = []  # initialise variable for voxel vals from direct image data
for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
        if mask_uniformity[i, j] == 1:  # 80% area mask
            save_value1 = imdata_conv[i, j]  # for uniformity calculation
            save_value2 = imdata[i, j]  # for SNR calculation
            RoiVoxelVals1 = np.append(RoiVoxelVals1, save_value1)
            RoiVoxelVals2 = np.append(RoiVoxelVals2, save_value2)

Smax = np.max(RoiVoxelVals1)
Smin = np.min(RoiVoxelVals1)
uniformity_measure = 100 * (1 - ((Smax - Smin) / (Smax + Smin)))
non_uniformity_measure = 100 - uniformity_measure

print('Integral Uniformity, Uint = ', uniformity_measure.__round__(2), '%')
print('Non-Uniformity Measure, N = ', non_uniformity_measure.__round__(2), '%')

# GREYSCALE UNIFORMITY MAP
mean_pixel_value = np.mean(RoiVoxelVals1)
GUM = np.zeros(dims)  # Greyscale Uniformity Map
# assign each voxel according to its intensity relative to the mean pixel value
# outlined in IPEM Report 112
for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
        if mask[i, j] == 1:
            # >20%
            if imdata_conv[i, j] >= (1.2 * mean_pixel_value):
                GUM[i, j] = 1
            # +10% and +20%
            if (1.1 * mean_pixel_value) <= imdata_conv[i, j] < (1.2 * mean_pixel_value):
                GUM[i, j] = 0.75
            # -10% and +10%
            if (0.9 * mean_pixel_value) <= imdata_conv[i, j] < (1.1 * mean_pixel_value):
                GUM[i, j] = 0.5
            # -10% and -20 %
            if (0.8 * mean_pixel_value) <= imdata_conv[i, j] < (0.9 * mean_pixel_value):
                GUM[i, j] = 0.25
            # < -20%
            if imdata_conv[i, j] < (0.8 * mean_pixel_value):
                GUM[i, j] = 0

# Display GUM
plt.figure()
plt.imshow(GUM, cmap='gray')
cbar = plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.ax.set_yticklabels(['< -20%', '-10% to -20%',
                         '-10% to +10%', '+10% to 20%', '> +20%'])
plt.axis('off')
plt.title('Greyscale Uniformity Map; scaled relative to mean pixel value')
plt.savefig(path + '_GUM.png')
plt.show()

# SNR measure
factor = 0.66  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112
mean_phantom = np.mean(RoiVoxelVals2)  # mean signal from image data (not filtered!)

# auto detection of 4 x background ROI samples (one in each corner of background)
bground_ROI = mask * 0  # initialise image matrix
idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
rows = idx[0]
cols = idx[1]
min_row = np.min(rows)  # first row of phantom
max_row = np.max(rows)  # last row of phantom

min_col = np.min(cols)  # first column of phantom
max_col = np.max(cols)  # last column of phantom

half_row = int(dims[0] / 2)  # half way row
mid_row1 = int(round(min_row / 2))
mid_row2 = int(round((((dims[0] - max_row) / 2) + max_row)))

half_col = int(dims[1] / 2)  # half-way column
mid_col1 = int(round(min_col / 2))
mid_col2 = int(round((((dims[1] - max_col) / 2) + max_col)))

print(half_col, half_row, mid_row1, mid_row2, mid_col1, mid_col2)

bROI1 = bground_ROI.copy()  # initialise image matrix for each corner ROI
bROI2 = bground_ROI.copy()
bROI3 = bground_ROI.copy()
bROI4 = bground_ROI.copy()

# 2 ROIs along each frequency and phase encoding direction
rr1, cc1 = ellipse(mid_row1, half_col, mid_row1, max_col - half_col)
bROI1[rr1, cc1] = 1

rr2, cc2 = ellipse(half_row, mid_col1, half_row - min_row, min_col - mid_col1)
bROI2[rr2, cc2] = 1

rr3, cc3 = ellipse(half_row, mid_col2, half_row - min_row, mid_col2 - max_col)
bROI3[rr3, cc3] = 1

rr4, cc4 = ellipse(mid_row2, half_col, dims[0] - mid_row2, half_col - min_col)
bROI4[rr4, cc4] = 1

# https://github.com/aaronfowles/breast_mri_qa/blob/master/breast_mri_qa/measure.py
ROIs = [bROI1, bROI2, bROI3, bROI4]
# erode each corner ROI to 10% of area of phantom ROI so total bground ROI is = 40% of signal ROI
# this is a completely arbitrary choice!
for region in ROIs:
    region_area = np.sum(region)
    roi_proportion = 0.2  # 5%
    target_roi_area = roi_proportion * new_mask_area  # based on phantom ROI area
    actual_roi_proportion = 1
    roi = region.copy()  # this is eroded variable in while loop
    while actual_roi_proportion > roi_proportion:
        roi = ndimage.binary_erosion(roi).astype(int)
        actual_roi_proportion = np.sum(roi) / float(new_mask_area)  # based on phantom ROI area
    bground_ROI = bground_ROI + roi  # append each updated corner ROI

# display the background noise ROI and signal ROI for the SNR calculation
plt.figure()
plt.subplot(121)
plt.imshow(bground_ROI)
plt.title('Background (Noise) ROI')
plt.axis('off')
plt.subplot(122)
plt.imshow(mask_uniformity)
plt.title('Signal ROI')
plt.axis('off')
plt.savefig(path + '_SNR_masks.png')
plt.show()

# background/noise voxel values
BGrndVoxelVals = []
for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
        if bground_ROI[i, j] == 1:
            save_value = imdata[i, j]
            BGrndVoxelVals = np.append(BGrndVoxelVals, save_value)

stdev_background = np.std(BGrndVoxelVals)  # noise = standard deviation of background voxel values

b_ground_samples = np.shape(BGrndVoxelVals)
b_ground_samples = b_ground_samples[0]
signal_samples = np.shape(RoiVoxelVals2)
signal_samples = signal_samples[0]
# check what % of signal ROI that background ROI is - should be approx 20%
print('Background ROI size is ', round((b_ground_samples / signal_samples) * 100, 2), '% of signal ROI size.')

##########
noise_mask = ndimage.binary_dilation(mask, iterations=2)  # dilate mask to avoid edge effects when displaying noise
noise_image = imdata * ~noise_mask  # image noise (phantom signal is masked out)
ALLBGrndVoxelVals = noise_image[noise_mask == 0]  # voxel values from all of background

# display noise image
plt.figure()
plt.imshow(noise_image, cmap='gray')
plt.axis('off')
plt.savefig(path + '_noise_image.png')
plt.show()

# histogram of noise to check that is follows non-Gaussian distribution
plt.figure()
plt.hist(ALLBGrndVoxelVals)
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.suptitle('Histogram of Background Noise')
plt.title('Demonstration of Non-Gaussian Distribution')
plt.savefig(path + '_noise_histogram.png')
plt.show()
##########

# SNR calculation
SNR_background = (factor * mean_phantom) / stdev_background
print('SNR = ', SNR_background.round(2))

# END
