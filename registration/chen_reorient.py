import argparse
from glob import glob
import os
import numpy as np
import itk
import shutil

def respacing(image):
    respacing_filter = itk.ResampleImageFilter.New(image)
    #### respacing
    ori_spacing = image.GetSpacing()
    ori_origin = image.GetOrigin()
    # output_origin = [0.0, 0.0, 0.0]  # 输出图像的原点
    output_spacing = [1.0, 1.0, 1.0]  # 输出图像的间距
    output_size = [image.GetLargestPossibleRegion().GetSize()[0],
                    image.GetLargestPossibleRegion().GetSize()[1],
                    image.GetLargestPossibleRegion().GetSize()[2]]
    # print(output_size)
    output_size = [int(output_size[k]*ori_spacing[k]/output_spacing[k]) for k in range(3)]
    # pad_size = (240, 240, 155)
    # output_origin = [ori_origin[k] + (output_size[k]-pad_size[k]) // 2 for k in range(3)] 

    respacing_filter.SetOutputOrigin(ori_origin)
    respacing_filter.SetOutputDirection(image.GetDirection())
    respacing_filter.SetSize(output_size)
    respacing_filter.SetOutputSpacing(output_spacing)
    
    respacing_filter.Update()
    respaced = respacing_filter.GetOutput()
    return respaced


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


if __name__ == '__main__':
    image_folder = "/ailab/user/chenlingzhi/data/brain_xh/tumor_nifti/NNZ/norm_data"
    output_folder = "/ailab/user/chenlingzhi/data/brain_xh/tumor_nifti_LPI/NNZ"
    phase_name = [ "DWI0.nii.gz", "T1.nii.gz", "T1-c.nii.gz", "T2.nii.gz", "FLAIR.nii.gz"]
    filenames = []
    for phase in phase_name:
        filenames += glob(os.path.join(image_folder, '*', phase))
    print(len(filenames))

    # dimension_type = set()
    for filename in sorted(filenames):
        pid = filename.split('/')[-2]
        if not os.path.exists(os.path.join(output_folder, pid)):
            os.mkdir(os.path.join(output_folder, pid))
        
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]

        if os.path.isfile(os.path.join(output_folder, pid, basename_wo_ext + '.nii.gz')):
            continue
        
        try:
            image = itk.imread(filename)
            
        except:
            print(filename)
            continue

        reoriented = reorient_to_rai(image)
       
        # reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64))
        reoriented.SetDirection(m)
        reoriented.Update()

        respaced = respacing(reoriented)
        # dimension = [str(respaced.GetLargestPossibleRegion().GetSize()[k]) for k in range(3)]
        # dimension_type.add(','.join(dimension))
        
        
        itk.imwrite(respaced, os.path.join(output_folder, pid, basename_wo_ext + '.nii.gz'))