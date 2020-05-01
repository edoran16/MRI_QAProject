from DICOM_test import dicom_read_and_write
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line, label, regionprops
from skimage import filters, segmentation
from skimage.morphology import binary_erosion

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

# SNR measure

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

### NEED TO APPLY LOW PASS FILTER TO IMAGE HERE!!!!

mask_uniformity = ROIerode
RoiVoxelVals = []
for i in np.linspace(0, dims[0]-1, dims[0], dtype=int):
    for j in np.linspace(0, dims[1]-1, dims[1], dtype=int):
        if mask_uniformity[i, j] == 1:
            save_value = imdata[i, j]
            RoiVoxelVals = np.append(RoiVoxelVals, save_value)

Smax = np.max(RoiVoxelVals)
Smin = np.min(RoiVoxelVals)
uniformity_measure = 100 * (1-((Smax-Smin)/(Smax+Smin)))
non_uniformity_measure = 100-uniformity_measure

print('Integral Uniformity, Uint = ', uniformity_measure.__round__(2), '%')
print('Non-Uniformity Measure, N = ', non_uniformity_measure.__round__(2), '%')

### MAKE GREYSCALE COLOUR MAP HERE!!!!
