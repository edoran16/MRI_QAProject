"""
====================
Read MagNET DICOM data
====================

"""

import os
import pydicom

# fetch the path to the test data
directpath = "MagNET_acceptance_test_data/scans/"

# list all folders in scans folder
with os.scandir(directpath) as it:
    for entry in it:
        pathtodir = "{0}{1}{2}".format(directpath, entry.name, '/resources/')
        with os.scandir(pathtodir) as iv:
            for fold in iv:
                if fold.name != 'secondary' and fold.name != 'SNAPSHOTS':
                    pathtofolder = "{0}{1}{2}".format(pathtodir, fold.name, '/files/')
                    print('Path to the DICOM directory: {}'.format(pathtofolder))
                    with os.scandir(pathtofolder) as iw:
                        for file in iw:
                            pathtofile = "{0}{1}".format(pathtofolder, file.name)
                            # load the data
                            dicom_file = pydicom.dcmread(pathtofile)
