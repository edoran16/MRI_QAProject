from pyxnat import Interface
import os
import glob

clear_folder = True
access_central = True
access_gu = False

if clear_folder:
    files = glob.glob('pyxnat_files/*')
    for f in files:
        print('Removing', f, '...')
        os.remove(f)

if access_central:
    central = Interface(server='https://central.xnat.org', user='edoran16', password='Westwood_1306')

    # Using pyxnat's object methods to walk down the path.
    # https://wiki.xnat.org/workshop-2016/files/29034956/29034952/1/1465403228621/Pyxnat+101.pdf
    central_project = central.select.project('M_A_T')
    print('Project:', central_project.label())
    central_subject = central_project.subject('MAGNETOM_Sola')
    print('Subject:', central_subject.label())
    central_experiment = central_subject.experiment('MAGNETOM_Sola_PR_1')
    print('Experiment:', central_experiment.label())

    the_scans = central_experiment.scans()

    for scan in the_scans:
        dicom_resource = scan.resource('DICOM')

        the_files = dicom_resource.files()
        store_file = 'C:/Users/GU Student/PycharmProjects/MRI_QAProject/pyxnat_files'
        for f in the_files:
            print('Getting', f.label(), 'and saving to', store_file)
            f.get_copy(os.path.join(store_file, f.label()))

if access_gu:
    print('Now for GU')

    gu = Interface(server='https://130.209.143.85', user='emma', password='Medphys_0520', verify=False)
    gu_project = gu.select.project('emmatest')
    gu_subject = gu_project.subject('phantom1')
    gu_experiment = gu_subject.experiment('BasicImage')

    the_gu_scans = gu_experiment.scans()
    for gscan in the_gu_scans:
        dicom_gu_resource = gscan.resource('DICOM')

        the_gfiles = dicom_gu_resource.files()
        store_gfile = 'C:/Users/GU Student/PycharmProjects/MRI_QAProject/pyxnat_files/'
        for gf in the_gfiles:
            print('Getting', gf.label(), 'and saving to', store_gfile)
            gf.get_copy(os.path.join(store_gfile, gf.label()))

