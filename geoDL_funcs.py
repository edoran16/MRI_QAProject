from pylab import *
import cv2
import os
from DICOM_test import dicom_read_and_write
from nibabel.viewers import OrthoSlicer3D
import numpy as np


def sort_import_data(directpath, geometry):
    with os.scandir(directpath) as the_folders:
        for folder in the_folders:
            fname = folder.name
            if re.search('-GEO_', fname):
                if re.search(geometry, fname):
                    print('Loading ...', fname)
                    folder = fname
                    pathtodicom = "{0}{1}{2}".format(directpath, folder, '/resources/DICOM/files/')

                    with os.scandir(pathtodicom) as it:
                        for file in it:
                            path = "{0}{1}".format(pathtodicom, file.name)

                    ds, imdata, df, dims = dicom_read_and_write(path)  # function from DICOM_test.py

                    try:
                        xdim, ydim = dims
                        print('Matrix Size =', xdim, 'x', ydim)

                        img = ((imdata / np.max(imdata)) * 255).astype('uint8')  # grayscale

                        cv2.imshow('dicom imdata', img)
                        cv2.waitKey(0)

                    except ValueError:
                        print('DATA INPUT ERROR: this is 3D image data')
                        OrthoSlicer3D(imdata).show()  # look at 3D volume data
                        sys.exit()

    return img, imdata



