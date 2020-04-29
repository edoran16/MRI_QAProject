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
    with open(path + ".txt", "w") as f:
        print(fulldicomfile, file=f)
    # assign image data
    imagedata = fulldicomfile.pixel_array
    imagedimensions = imagedata.shape
    return fulldicomfile, imagedata, imagedimensions


directpath = "data_to_get_started/single_slice_dicom/"
filename = "image1"
path = "{0}{1}".format(directpath, filename)
ds, imdata, dims = dicom_read_and_write(path)

# display image
plt.figure(figsize=[100, 100])
plt.imshow(imdata, cmap='bone')
plt.colorbar()
plt.savefig(path + '.png')
plt.show()

# draw line profile
src = (200, 200)
dst = (5, 5)
linewidth = 2
output = profile_line(imdata, src, dst)

plt.figure(figsize=[100, 100])
plt.plot(output)
plt.show()

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

improfile = imdata
improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 0

plt.figure(figsize=[100, 100])
plt.imshow(improfile, cmap='bone')
plt.colorbar()
plt.savefig(path + '.png')
plt.show()

# SNR measure





