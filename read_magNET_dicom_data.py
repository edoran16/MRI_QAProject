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
        # print(entry.name)
        pathtodir = "{0}{1}{2}".format(directpath, entry.name, '/resources/')
        # print(pathtodir)
        with os.scandir(pathtodir) as iv:
            for fold in iv:
                if fold.name != 'secondary' and fold.name != 'SNAPSHOTS':
                    # print(fold.name)
                    pathtofolder = "{0}{1}{2}".format(pathtodir, fold.name, '/files/')
                    # print(pathtofolder)
                    print('Path to the DICOM directory: {}'.format(pathtofolder))
                    with os.scandir(pathtofolder) as iw:
                        for file in iw:
                            # print(file.name)
                            pathtofile = "{0}{1}".format(pathtofolder, file.name)
                            # print(pathtofile)
                            # load the data
                            dicom_file = pydicom.dcmread(pathtofile)
                            # base_dir = dirname(pathtofile)
