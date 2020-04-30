import pydicom

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
