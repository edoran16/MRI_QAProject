import pydicom
import matplotlib.pyplot as plt

# get test data
direct = "data_to_get_started/single_slice_dicom/"
filename = "image1"
path = direct + filename
ds = pydicom.dcmread(path)

# export metadata into output text file to see all entries
with open(path + ".txt", "w") as f:
    print(ds, file=f)

# assign image data
imdata = ds.pixel_array
dims = imdata.shape

# display image
plt.figure(figsize=[100, 100])
plt.imshow(imdata, cmap='bone')
plt.savefig(path + '.png')
plt.show()





