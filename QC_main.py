from QC_mod import phantomimage_dicom, explore_noise, create_report, MagNETdata_dicom


def main():

    gen = phantomimage_dicom()  # generic phantom object (2D)
    dicomfile, single_slice_im, imdims = gen.dicom_read_and_write(False)

    src = (imdims[0] / 2, 0 + 1)  # starting point
    dst = (imdims[0] / 2, imdims[1] - 1)  # finish point

    line_vals = gen.return_line_profile(single_slice_im, src, dst, 2, False)

    mask = gen.foreground_detection(single_slice_im, False)

    ROIerode, old_mask_area, new_mask_area = gen.create_80percentROI(mask, single_slice_im.ndim, plotflag=False)

    imLPF = gen.apply_LPF(single_slice_im, plotflag=False)  # default LPF kernel used

    Uint, voxelvals = gen.calculate_Uint(ROIerode, imLPF)  # integral uniformity calculation

    gen.create_GUM(voxelvals, mask, imLPF, False)

    bgROI, BGvals = gen.background_ROI_detection(imLPF, ROIerode, mask, plotflag=False)  # default roi proportion used

    SNR = gen.calculate_SNR(voxelvals, BGvals)  # default factor used

    explore_noise(mask, single_slice_im, gen.pathtofile, plotflag=False)

    create_report(SNR, Uint)


if __name__ == '__main__':
    #main()
    gen = MagNETdata_dicom()  # MagNET data class
    dicomfile, single_slice_im, df, imdims = gen.dicom_read_and_write()

    xdim, ydim, zdim = imdims






