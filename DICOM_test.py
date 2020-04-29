import pydicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line


def dicom_read_and_write(pathtofile):
    """ function to read dicom file from specified path
    :param pathtofile: full path to dicom file
    :return: returns dicom file, image data and image dimensions
    """

    # get test data
    fulldicomfile = pydicom.dcmread(pathtofile)
    # export metadata into output text file to see all entries
    #with open(path + ".txt", "w") as f:
        #print(fulldicomfile, file=f)
    # assign image data
    imagedata = fulldicomfile.pixel_array
    imagedimensions = imagedata.shape
    return fulldicomfile, imagedata, imagedimensions


directpath = "data_to_get_started/single_slice_dicom/"
filename = "image1"
path = "{0}{1}".format(directpath, filename)
ds, imdata, dims = dicom_read_and_write(path)

# display image
# plt.figure(figsize=[100, 100])
# plt.imshow(imdata, cmap='bone')
# plt.colorbar()
# plt.savefig(path + '.png')
# plt.show()

# draw line profile
src = (200, 200)
dst = (5, 5)
linewidth = 2
output = profile_line(imdata, src, dst)

# plt.figure(figsize=[100, 100])
# plt.plot(output)
# plt.show()

# display profile line on phantom
src_row, src_col = src = np.asarray(src, dtype=float)
dst_row, dst_col = dst = np.asarray(dst, dtype=float)
d_row, d_col = dst - src
theta = np.arctan2(d_row, d_col)

length = int(np.ceil(np.hypot(d_row, d_col) + 1))
# we add one above because we include the last point in the profile
# (in contrast to standard numpy indexing)
line_col = np.linspace(src_col, dst_col, length)
line_row = np.linspace(src_row, dst_row, length)

# we subtract 1 from linewidth to change from pixel-counting
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

# plt.figure(figsize=[100, 100])
# plt.imshow(improfile, cmap='bone')
# plt.colorbar()
# plt.savefig(path + '_line_profile.png')
# plt.show()

# SNR measure

# UNIFORMITY MEASURE
# OTSU THRESHOLD
from skimage import filters, segmentation
val = filters.threshold_otsu(imdata)
mask = imdata < val
mask = ~mask

clean_border = segmentation.clear_border(mask).astype(np.int)
phantom_edges = segmentation.mark_boundaries(imdata, clean_border)
phantom_edges2 = segmentation.find_boundaries(mask, mode='thin').astype(np.uint8)

plt.figure(figsize=(8, 3.5))
plt.subplot(131)
plt.imshow(mask)
plt.axis('off')
plt.subplot(132)
plt.imshow(clean_border)
plt.axis('off')
plt.subplot(133)
plt.imshow(phantom_edges2)
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=[100, 100])
plt.imshow(mask)
plt.savefig(path + '_otsumask.png')
plt.show()

# LABEL OTSU
from skimage.measure import label, regionprops

label_img, num = label(mask, connectivity=imdata.ndim, return_num=True)

props = regionprops(label_img)

centre_point = np.round(props[0].centroid)
area100 = props[0].area
r100 = int(np.round(np.sqrt(area100/np.pi)))

area80 = area100*0.8
r80 = np.round(np.sqrt(area80/np.pi))
r80 = np.round(r80)
r80 = int(r80)

cx, cy = centre_point # The center of circle
cx = int(cx)
cy = int(cy)

ROI80 = np.zeros(mask.shape)
y80, x80 = np.ogrid[-r80: r80, -r80: r80]
index80 = x80**2 + y80**2 <= r80**2
ROI80[cy-r80:cy+r80, cx-r80:cx+r80][index80] = 1

ROI100 = np.zeros(mask.shape)
y100, x100 = np.ogrid[-r100: r100, -r100: r100]
index100 = x100**2 + y100**2 <= r100**2
ROI100[cy-r100:cy+r100, cx-r100:cx+r100][index100] = 1

plt.figure(figsize=[100, 100])
plt.subplot(1, 4, 1)
plt.imshow(imdata)
plt.subplot(1, 4, 2)
plt.imshow(mask)
plt.subplot(1, 4, 3)
plt.imshow(ROI100+mask)
plt.subplot(1, 4, 4)
plt.imshow(ROI80+mask)
plt.show()

