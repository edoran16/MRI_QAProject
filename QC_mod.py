from __future__ import print_function
from skimage.measure import profile_line, label, regionprops
from skimage import filters, segmentation
from skimage.morphology import binary_erosion
from scipy import ndimage
from skimage.draw import ellipse
from scipy import stats
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

# https://github.com/ccipd/MRQy/blob/master/QCF.py
# Classes and object theory from O'Reilly: Effective Computation in Physics. A.Scopatz & K. D. Huff

class phantomimage_dicom:
    """ Class for dicom single slice phantom image.
     Attributes
     ----------
     Dimensions
     Integral Uniformity
     SNR"""

    def __init__(self):
        # TODO: this section will need changed to give correct path to any future data i.e. in XNAT or Prisma files
        self.directpath = "data_to_get_started/single_slice_dicom/"  # path to DICOM file
        self.filename = "image1"
        self.pathtofile = "{0}{1}".format(self.directpath, self.filename)
        print("Init called!")

    def dicom_read_and_write(self, plotflag: object = False) -> object:
        """ function to read dicom file from specified path
        :param pathtofile: full path to dicom file
        :return: returns dicom file, image data and image dimensions
        """
        # get test data
        fulldicomfile = pydicom.dcmread(self.pathtofile)
        # export metadata into output text file to see all entries
        with open(self.pathtofile + ".txt", "w") as f:
            print(fulldicomfile, file=f)
            # TODO: change this to saving metadata as dataframe
        # assign image data
        imagedata = fulldicomfile.pixel_array
        imagedimensions = imagedata.shape

        # pandas data frame for meta data
        # https://stackoverflow.com/questions/56601525/how-to-store-the-header-data-of-a-dicom-file-in-a-pandas-dataframe
        df = pd.DataFrame(fulldicomfile.values())
        df[0] = df[0].apply(
            lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
        df['name'] = df[0].apply(lambda x: x.name)
        df['value'] = df[0].apply(lambda x: x.value)
        df = df[['name', 'value']]

        # display image
        if plotflag:
            plt.figure()
            plt.imshow(imagedata, cmap='bone')
            plt.colorbar()
            plt.axis('off')
            plt.show()

        return fulldicomfile, imagedata, df, imagedimensions



    def return_line_profile(self, imdata, src, dst, lw, plotflag=False):
        """ Draw line profile across centre line of phantom
        Specify source (src) and destination (dst) coordinates
        This function produces plot of voxel values and image of sampled line on phantom.
        Linewidth = width of the line (mean taken over width)
        Return: = output. Which is the sampled voxel values. """

        linewidth = lw
        output = profile_line(imdata, src, dst)  # voxel values along specified line

        # plot profile line output vs. voxel sampled
        if plotflag:
            plt.figure()
            plt.plot(output)
            plt.xlabel('Voxels')
            plt.ylabel('Signal')
            plt.show()

        # display profile line on phantom: from source code of profile_line function
        src_row, src_col = src = np.asarray(src, dtype=float)
        dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)

        length = int(np.ceil(np.hypot(d_row, d_col) + 1))
        # add one above to include the last point in the profile
        # (in contrast to standard numpy indexing)
        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)

        # subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        col_width = (linewidth - 1) * np.sin(-theta) / 2
        row_width = (linewidth - 1) * np.cos(theta) / 2
        perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                          linewidth) for row_i in line_row])
        perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                          linewidth) for col_i in line_col])

        improfile = np.copy(imdata)
        improfile[np.array(np.round(perp_rows), dtype=int), np.array(np.round(perp_cols), dtype=int)] = 0

        # plot sampled line on phantom to visualise where output comes from
        if plotflag:
            plt.figure()
            plt.imshow(improfile, cmap='bone')
            plt.colorbar()
            plt.axis('off')
            plt.show()

        return output


    def foreground_detection(self, imdata, plotflag=False):
        val = filters.threshold_otsu(imdata)  # OTSU threshold to segment phantom
        mask = imdata > val  # phantom mask
        # TODO: update threshold method for alternative foreground detection

        phantom_edges = segmentation.find_boundaries(mask, mode='thin').astype(np.uint8)  # finds outline of mask

        if plotflag:
            plt.figure()
            plt.subplot(131)
            plt.imshow(mask)
            plt.title('Otsu Mask')
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(phantom_edges)
            plt.title('Phantom Boundary')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(mask + phantom_edges)
            plt.axis('off')
            plt.title('Overlay')
            plt.tight_layout()
            plt.show()

        return mask

    def create_80percentROI(self, mask, nodims, plotflag=False):
        """ create mask that encompassess 80% of foreground mask
        Mask = original 100% area mask
        nodims = number of dimensions of the image data (2D in this case!!)
        Function returns the eroded mask, original and updated area (in terms of number of voxels)"""

        label_img, num = label(mask, connectivity=nodims, return_num=True)  # labels the mask
        props = regionprops(label_img)  # returns region properties for phantom mask ROI
        area100 = props[0].area  # area of phantom mask
        area80 = area100 * 0.8  # desired phantom area = 80% of phantom area [IPEM Report 112]

        temp_mask = np.copy(mask)  # copy of phantom mask
        old_mask_area = np.sum(temp_mask)  # 100% area

        new_mask_area = old_mask_area  # initialise new_mask_area to be updated in while loop
        count = 0  # initialise counter for while loop
        while new_mask_area > area80:  # whilst the area is greater than 80% of original area continue eroding mask
            count = count + 1  # counter
            shrunk_mask = binary_erosion(temp_mask)
            unraveled_mask = np.ravel(shrunk_mask)
            new_mask_area = np.sum(unraveled_mask)
            temp_mask = np.reshape(unraveled_mask, mask.shape)

        ROIerode = temp_mask  # eroded mask from while loop

        if plotflag:
            plt.figure()
            plt.subplot(121)
            plt.imshow(mask)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(mask * ~ROIerode)
            plt.axis('off')
            plt.show()

        return ROIerode, old_mask_area, new_mask_area

    def apply_LPF(self, imdata, kernel=(1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), plotflag=False):
        """ LOW PASS FILTER APPLIED TO REDUCE EFFECTS OF NOISE
        specified as pre-processing step in IPEM Report 112.
        Input = image and kernel (default kernel is as specified in IPEM Report 12 ^^)
        Output = filtered image"""
        # TODO: confirm that this only applies to low SNR data? How to call this function when required

        a = imdata.copy()  # copy of DICOM image
        imdata_conv = ndimage.convolve(a, kernel, mode='constant', cval=0.0)  # convolution of image and kernel

        # display image and filtered image (normalised)
        if plotflag:
            plt.figure()
            plt.subplot(121)
            plt.imshow(imdata / np.max(imdata), cmap='bone')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(imdata_conv / np.max(imdata_conv), cmap='bone')
            plt.title('Low Pass Filtered Image')
            plt.axis('off')
            plt.show()

        return imdata_conv

    def calculate_Uint(self, mask_uniformity, image):
        """Integral Uniformity calculation according to IPEM Report 112.
        mask_uniformity =  mask for measuring uniformity is the eroded mask
        image = original or filtered image
        Output = integral uniformity result"""

        RoiVoxelVals = []  # initialise variable for voxel vals from image data
        dims = mask_uniformity.shape
        for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
            for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
                if mask_uniformity[i, j] == 1:  # 80% area mask
                    save_value = image[i, j]
                    RoiVoxelVals = np.append(RoiVoxelVals, save_value)

        Smax = np.max(RoiVoxelVals)
        Smin = np.min(RoiVoxelVals)
        uniformity_measure = 100 * (1 - ((Smax - Smin) / (Smax + Smin)))

        return uniformity_measure, RoiVoxelVals

    def create_GUM(self, RoiVoxelVals, mask, image, plotflag=True):
        """Produce greyscale uniformity map.
        Categories of uniformity value defined according to IPEM Report 112."""
        # GREYSCALE UNIFORMITY MAP
        mean_pixel_value = np.mean(RoiVoxelVals)
        dims = mask.shape
        GUM = np.zeros(dims)  # Greyscale Uniformity Map
        # assign each voxel according to its intensity relative to the mean pixel value
        # outlined in IPEM Report 112
        for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
            for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
                if mask[i, j] == 1:
                    # >20%
                    if image[i, j] >= (1.2 * mean_pixel_value):
                        GUM[i, j] = 1
                    # +10% and +20%
                    if (1.1 * mean_pixel_value) <= image[i, j] < (1.2 * mean_pixel_value):
                        GUM[i, j] = 0.75
                    # -10% and +10%
                    if (0.9 * mean_pixel_value) <= image[i, j] < (1.1 * mean_pixel_value):
                        GUM[i, j] = 0.5
                    # -10% and -20 %
                    if (0.8 * mean_pixel_value) <= image[i, j] < (0.9 * mean_pixel_value):
                        GUM[i, j] = 0.25
                    # < -20%
                    if image[i, j] < (0.8 * mean_pixel_value):
                        GUM[i, j] = 0

        # Display GUM
        if plotflag:
            plt.figure()
            plt.imshow(GUM, cmap='gray')
            cbar = plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
            cbar.ax.set_yticklabels(['< -20%', '-10% to -20%',
                                 '-10% to +10%', '+10% to 20%', '> +20%'])
            plt.axis('off')
            plt.title('Greyscale Uniformity Map; scaled relative to mean pixel value')
            plt.show()

        return

    def background_ROI_detection(self, image, ROIerode, mask, roi_proportion=0.2, plotflag=False):
        """Create 4 x ROI in background of iamge for SNR calculation
        image = filtered (or unfiltered image)
        ROIerode = 80% area mask
        mask = 100% area mask
        roi_proportion = multiply this by 4 to get fraction of ROIerode area that total background ROI area will be
        default roi_proportion = 0.2  >> 0.2*4 = 0.8. Total background ROI is 80% of signal ROI"""
        # auto detection of 4 x background ROI samples (one in each corner of background)
        dims = mask.shape  # dimensions of mask
        new_mask_area = np.sum(ROIerode)
        bground_ROI = mask * 0  # initialise image matrix
        idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
        rows = idx[0]
        cols = idx[1]
        min_row = np.min(rows)  # first row of phantom
        max_row = np.max(rows)  # last row of phantom

        min_col = np.min(cols)  # first column of phantom
        max_col = np.max(cols)  # last column of phantom

        half_row = int(dims[0] / 2)  # half way row
        mid_row1 = int(round(min_row / 2))
        mid_row2 = int(round((((dims[0] - max_row) / 2) + max_row)))

        half_col = int(dims[1] / 2)  # half-way column
        mid_col1 = int(round(min_col / 2))
        mid_col2 = int(round((((dims[1] - max_col) / 2) + max_col)))

        bROI1 = bground_ROI.copy()  # initialise image matrix for each corner ROI
        bROI2 = bground_ROI.copy()
        bROI3 = bground_ROI.copy()
        bROI4 = bground_ROI.copy()

        # 2 ROIs along each frequency and phase encoding direction
        rr1, cc1 = ellipse(mid_row1, half_col, mid_row1, max_col - half_col)
        bROI1[rr1, cc1] = 1

        rr2, cc2 = ellipse(half_row, mid_col1, half_row - min_row, min_col - mid_col1)
        bROI2[rr2, cc2] = 1

        rr3, cc3 = ellipse(half_row, mid_col2, half_row - min_row, mid_col2 - max_col)
        bROI3[rr3, cc3] = 1

        rr4, cc4 = ellipse(mid_row2, half_col, dims[0] - mid_row2, half_col - min_col)
        bROI4[rr4, cc4] = 1

        # https://github.com/aaronfowles/breast_mri_qa/blob/master/breast_mri_qa/measure.py
        ROIs = [bROI1, bROI2, bROI3, bROI4]
        # erode each corner ROI to 10% of area of phantom ROI so total bground ROI is = 40% of signal ROI
        # this is a completely arbitrary choice!
        for region in ROIs:
            actual_roi_proportion = 1
            roi = region.copy()  # this is eroded variable in while loop
            while actual_roi_proportion > roi_proportion:
                roi = ndimage.binary_erosion(roi).astype(int)
                actual_roi_proportion = np.sum(roi) / float(new_mask_area)  # based on phantom ROI area
            bground_ROI = bground_ROI + roi  # append each updated corner ROI

        # display background noise ROI and signal ROI for SNR calculation
        if plotflag:
            plt.figure()
            plt.imshow(bground_ROI)
            plt.title('Background (Noise) ROI')
            plt.axis('off')
            plt.show()

        # background/noise voxel values
        BGrndVoxelVals = []
        for i in np.linspace(0, dims[0] - 1, dims[0], dtype=int):
            for j in np.linspace(0, dims[1] - 1, dims[1], dtype=int):
                if bground_ROI[i, j] == 1:
                    save_value = image[i, j]
                    BGrndVoxelVals = np.append(BGrndVoxelVals, save_value)

        return bground_ROI, BGrndVoxelVals

    def calculate_SNR(self, SignalVals, BgVals, factor=0.66):
        """SNR calculation (from single image, noise estimation from background).
        Default factor = 0.66 (for single element coil, background noise follows Rayleigh distribution IPEM Report 112)"""

        mean_phantom = np.mean(SignalVals)  # mean signal from image data (filtered or unfiltered)
        stdev_background = np.std(BgVals)  # noise = standard deviation of background voxel values
        SNR_background = (factor * mean_phantom) / stdev_background

        return SNR_background


class MagNETdata_dicom:
    """ Class for MagNET acceptance testing data."""

    def __init__(self):
        # TODO: this section will need changed to give correct path to any future data i.e. in XNAT or Prisma files
        self.directpath = "MagNET_acceptance_test_data/scans/"
        self.folder = "45-SPINE_123/resources/DICOM/files/"
        self.filename = "1.3.12.2.1107.5.2.51.182690.30000019050607313408100000039-45-1-wm8zff.dcm"
        self.path = "{0}{1}{2}".format(self.directpath, self.folder, self.filename)
        print("Init called!")

    def dicom_read_and_write(self):
        """ function to read dicom file from specified path
        :param pathtofile: full path to dicom file
        :return: returns dicom file, image data and image dimensions
        """
        # get test data
        fulldicomfile = pydicom.dcmread(self.path)
        # assign image data
        imagedata = fulldicomfile.pixel_array
        imagedimensions = imagedata.shape

        # pandas data frame for meta data
        # https://stackoverflow.com/questions/56601525/how-to-store-the-header-data-of-a-dicom-file-in-a-pandas-dataframe
        df = pd.DataFrame(fulldicomfile.values())
        df[0] = df[0].apply(
            lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
        df['name'] = df[0].apply(lambda x: x.name)
        df['value'] = df[0].apply(lambda x: x.value)
        df = df[['name', 'value']]

        return fulldicomfile, imagedata, df, imagedimensions


def explore_noise(mask, imdata, path, plotflag=False):
    noise_mask = ndimage.binary_dilation(mask, iterations=2)  # dilate mask to avoid edge effects when displaying noise
    noise_image = imdata * ~noise_mask  # image noise (phantom signal is masked out)
    ALLBGrndVoxelVals = noise_image[noise_mask == 0]  # voxel values from all of background

    sz = ALLBGrndVoxelVals.shape

    nobins = 50

    # display noise image
    if plotflag:
        # noise with signal masked out
        plt.figure()
        plt.imshow(noise_image, cmap='gray')
        plt.axis('off')
        plt.savefig(path + '_noise_phantom_masked.png', bbox_inches='tight')
        plt.show()
        # noise with adjusted contrast in order to see noise
        plt.figure()
        plt.imshow(imdata, cmap='gray', vmin=0, vmax=50)
        plt.axis('off')
        plt.savefig(path + '_noise_phantom_windowed.png', bbox_inches='tight')
        plt.show()

        # histogram of noise to check that is follows non-Gaussian distribution
        plt.figure()
        plt.hist(ALLBGrndVoxelVals, nobins)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Number of Pixels')
        plt.suptitle('Histogram of Background Noise')
        plt.title('Demonstration of Non-Gaussian Distribution')
        plt.savefig(path + '_noise_histogram.png', bbox_inches='tight')
        plt.show()

        # fit to histogram
        plt.figure()
        # plot histogram PDF >> http://danielhnyk.cz/fitting-distribution-histogram-using-python/
        plt.hist(ALLBGrndVoxelVals, 30, density=True)
        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)

        lnspc = np.linspace(0, xmax, len(ALLBGrndVoxelVals))  # xmin should be zero

        # lets try the normal distribution first
        m1, s = stats.norm.fit(ALLBGrndVoxelVals)  # get mean and standard deviation
        pdf_g = stats.norm.pdf(lnspc, m1, s)  # now get theoretical values in our interval
        plt.plot(lnspc, pdf_g, label="Norm")  # plot it

        # guess what :)
        m2, v = stats.rayleigh.fit(ALLBGrndVoxelVals)
        pdf_ray = stats.rayleigh.pdf(lnspc, m2, v)
        plt.plot(lnspc, pdf_ray, label="Rayleigh")
        plt.legend(['Normal', 'Rayleigh', 'Noise'])
        plt.xlim([0, np.max(ALLBGrndVoxelVals)])
        plt.xlabel('Noise Voxel Value')
        plt.ylabel('Frequency')
        plt.savefig(path + '_noise_histogram_fit.png', bbox_inches='tight')
        plt.show()


def create_report(SNR, Uint):
    r = {'SNR': [SNR, SNR], 'Uint': [Uint, Uint]}
    print(r)
    df = pd.DataFrame(r)
    print(df)
    print(df.SNR[0])
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('testreport.html')

    template_vars = {"title": "Quality Control- Basic Phantom Results",
                     "results_table": df.to_html(),
                     "TestText": "Single slice DICOM image of Phantom."}

    html_out = template.render(template_vars)

    #HTML(string='<img src="data_to_get_started/single_slice_dicom/image1.png">').write_pdf("testreport.pdf")
    #HTML(string="<p> Testing... </p>").write_pdf("testreport.pdf")
    HTML(string=html_out).write_pdf("testreport.pdf")

    return



