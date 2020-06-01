import pydicom
import pandas as pd

def dicom_read_and_write(pathtofile, writetxt=False):
    """ function to read dicom file from specified path
    :param pathtofile: full path to dicom file
    :return: returns dicom file, image data and image dimensions
    """

    # get test data
    fulldicomfile = pydicom.dcmread(pathtofile)
    # export metadata into output text file to see all entries
    if writetxt:
        with open(pathtofile + ".txt", "w") as f:
            print(fulldicomfile, file=f)
    # assign image data
    imagedata = fulldicomfile.pixel_array
    imagedimensions = imagedata.shape

    # pandas data frame for meta data
    # https://stackoverflow.com/questions/56601525/how-to-store-the-header-data-of-a-dicom-file-in-a-pandas-dataframe
    df = pd.DataFrame(fulldicomfile.values())
    df[0] = df[0].apply(lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
    df['name'] = df[0].apply(lambda x: x.name)
    df['value'] = df[0].apply(lambda x: x.value)
    df = df[['name', 'value']]

    return fulldicomfile, imagedata, df, imagedimensions
