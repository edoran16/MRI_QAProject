import pydicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line, label, regionprops
from skimage import filters, segmentation
from skimage.morphology import binary_erosion
from scipy import ndimage

# https://github.com/ccipd/MRQy/blob/master/QCF.py

def dicom_read_and_write(pathtofile):
    """ function to read dicom file from specified path
    :param pathtofile: full path to dicom file
    :return: returns dicom file, image data and image dimensions
    """
    # get test data
    fulldicomfile = pydicom.dcmread(pathtofile)
    # export metadata into output text file to see all entries
    with open(pathtofile + ".txt", "w") as f:
        print(fulldicomfile, file=f)
    # assign image data
    imagedata = fulldicomfile.pixel_array
    imagedimensions = imagedata.shape
    return fulldicomfile, imagedata, imagedimensions