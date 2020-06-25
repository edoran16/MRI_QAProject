import geoDL_funcs as gf

directpath = "MagNET_acceptance_test_data/scans/"
imagepath = "MagNET_acceptance_test_data/GEO_Images/"

geos = ['_TRA', '_SAG', '_COR']

for ii in range(len(geos)):
    geometry = geos[ii]
    print('Data geometry =', geometry, '.')
    if geometry == '_TRA':
        caseT = True  # transverse
        caseS = False  # sagittal
        caseC = False  # coronal
    if geometry == '_SAG':
        caseT = False  # transverse
        caseS = True  # sagittal
        caseC = False  # coronal
    if geometry == '_COR':
        caseT = False  # transverse
        caseS = False  # sagittal
        caseC = True  # coronal

    img, imdata = gf.sort_import_data(directpath, geometry)
