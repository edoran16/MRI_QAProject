from DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line, label, regionprops
from skimage import filters, segmentation
from skimage.morphology import binary_erosion
from scipy import ndimage

directpath = "data_to_get_started/single_slice_dicom/"
filename = "image1"
path = "{0}{1}".format(directpath, filename)
ds, imdata, dims = dicom_read_and_write(path)

# display image
plt.figure()
plt.imshow(imdata, cmap='bone')
plt.colorbar()
plt.axis('off')
plt.savefig(path + '.png')
plt.show()

# draw line profile
src = (200, 200)  # starting point
dst = (5, 5)  # finish point
linewidth = 2  # width of the line (mean taken over width)
output = profile_line(imdata, src, dst)

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

plt.figure()
plt.imshow(improfile, cmap='bone')
plt.colorbar()
plt.axis('off')
plt.savefig(path + '_line_profile.png')
plt.show()

# UNIFORMITY MEASURE
# OTSU THRESHOLD
val = filters.threshold_otsu(imdata)
mask = imdata > val

phantom_edges = segmentation.find_boundaries(mask, mode='thin').astype(np.uint8)

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
plt.imshow(mask+phantom_edges)
plt.axis('off')
plt.title('Overlay')
plt.tight_layout()
plt.savefig(path + '_otsu_boundary_mask.png')
plt.show()

# LABEL OTSU
label_img, num = label(mask, connectivity=imdata.ndim, return_num=True)
# regionprops
props = regionprops(label_img)
area100 = props[0].area
area80 = area100*0.8  # desired phantom area = 80% of phantom area [IPEM Report 112]

ROIerode = []
first_mask = np.copy(mask)
old_mask_area = np.sum(first_mask)

new_mask_area = old_mask_area
count = 0
while new_mask_area > area80:
    count = count + 1
    print(count) # number of iterations to reduce mask to desired 80% area
    shrunk_mask = binary_erosion(first_mask)
    unraveled_mask = np.ravel(shrunk_mask)
    new_mask_area = np.sum(unraveled_mask)
    first_mask = np.reshape(unraveled_mask, dims)

print(new_mask_area/old_mask_area)
ROIerode = first_mask

plt.figure()
plt.subplot(121)
plt.imshow(mask)
plt.subplot(122)
plt.imshow(mask*~ROIerode)
plt.savefig(path + '_eroded_mask.png')
plt.show()

### APPLY LOW PASS FILTER TO REDUCE EFFECTS OF NOISE - this might not actually be necessary... since SNR is high

a = imdata.copy()
k = (1/16)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
imdata_conv = ndimage.convolve(a, k, mode='constant', cval=0.0)

plt.figure()
plt.subplot(121)
plt.imshow(imdata/np.max(imdata), cmap='bone')
plt.subplot(122)
plt.imshow(imdata_conv/np.max(imdata_conv), cmap='bone')
plt.savefig(path + '_convolution.png')
plt.show()

mask_uniformity = ROIerode
RoiVoxelVals1 = []
RoiVoxelVals2 = []
for i in np.linspace(0, dims[0]-1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1]-1, dims[1], dtype=int):
        if mask_uniformity[i, j] == 1:
            save_value1 = imdata_conv[i, j]  # for uniformity calculation
            save_value2 = imdata[i, j]  # for SNR calculation
            RoiVoxelVals1 = np.append(RoiVoxelVals1, save_value1)
            RoiVoxelVals2 = np.append(RoiVoxelVals2, save_value2)

Smax = np.max(RoiVoxelVals1)
Smin = np.min(RoiVoxelVals1)
uniformity_measure = 100 * (1-((Smax-Smin)/(Smax+Smin)))
non_uniformity_measure = 100-uniformity_measure

print('Integral Uniformity, Uint = ', uniformity_measure.__round__(2), '%')
print('Non-Uniformity Measure, N = ', non_uniformity_measure.__round__(2), '%')

# GREYSCALE UNIFORMITY MAP
mean_pixel_value = np.mean(RoiVoxelVals1)
GUM = np.zeros(dims)
for i in np.linspace(0, dims[0]-1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1]-1, dims[1], dtype=int):
        if mask[i, j] == 1:
            # >20%
            if imdata_conv[i, j] >= (1.2*mean_pixel_value):
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
factor = 0.66  # for single element coil, background noise follows Rayleigh distribution
mean_phantom = np.mean(RoiVoxelVals2)

# background ROI samples
bground_ROI = mask*0
idx = np.where(mask)
rows = idx[0]
cols = idx[1]
min_row = np.min(rows)
max_row = np.max(rows)

min_col_bool = np.isin(rows, min_row)
min_col_idx = np.where(min_col_bool)

min_col = cols[0]
max_col = cols[-1]

half_col = int(dims[1]/2)

bROI1 = bground_ROI.copy()
bROI2 = bground_ROI.copy()
bROI3 = bground_ROI.copy()
bROI4 = bground_ROI.copy()

bROI1[0:min_row, 0:half_col] = 1
bROI2[0:min_row, half_col:dims[1]] = 1
bROI3[max_row:dims[0], 0:half_col] = 1
bROI4[max_row:dims[0], half_col:dims[1]] = 1

# https://github.com/aaronfowles/breast_mri_qa/blob/master/breast_mri_qa/measure.py
ROIs = [bROI1, bROI2, bROI3, bROI4]
ROIs_new = ROIs.copy()
var = 0
for qq in ROIs:
    region = qq
    region_area = np.sum(region)
    roi_proportion = 0.25
    target_roi_area = roi_proportion * region_area
    actual_roi_proportion = 1
    roi = region.copy()
    while actual_roi_proportion > roi_proportion:
        roi = ndimage.binary_erosion(roi).astype(int)
        actual_roi_proportion = np.sum(roi) / float(region_area)
    ROIs_new[var] = roi
    bground_ROI = bground_ROI + roi
    var = var + 1

plt.figure()
plt.imshow(bground_ROI+mask_uniformity+mask)
plt.savefig(path + '_SNR_masks.png')
plt.show()

BGrndVoxelVals = []
for i in np.linspace(0, dims[0]-1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1]-1, dims[1], dtype=int):
        if bground_ROI[i, j] == 1:
            save_value = imdata[i, j]
            BGrndVoxelVals = np.append(BGrndVoxelVals, save_value)

stdev_background = np.std(BGrndVoxelVals)
print(BGrndVoxelVals.shape)

plt.figure()
plt.hist(BGrndVoxelVals)
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.savefig(path + '_noise_histogram.png')
plt.show()

SNR_background = (factor * mean_phantom)/stdev_background
print('SNR = ', SNR_background.round(2))

