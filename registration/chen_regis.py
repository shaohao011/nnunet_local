import ants
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm 
# import glob
import shutil
print("test...")

def do_registration(fixed_path, moving_path, save_path, mode='QuickRigid'):
    fix_itk = sitk.ReadImage(fixed_path)
    fix_np = sitk.GetArrayFromImage(fix_itk).astype('float32')
    mov_np = sitk.GetArrayFromImage(sitk.ReadImage(moving_path)).astype('float32')
    out = ants.registration(ants.from_numpy(fix_np), 
                    ants.from_numpy(mov_np), type_of_transform=mode)#'Syn
    # disp_np = ants.image_read(out['fwdtransforms'][0]).numpy() # 这是获得位移场
    # print(fix_np.shape, mov_np.shape)
    warp_np = out['warpedmovout'].numpy() # 这是变形后的图像
    warp_itk = sitk.GetImageFromArray(warp_np)
    warp_itk.SetDirection(fix_itk.GetDirection())
    warp_itk.SetOrigin(fix_itk.GetOrigin())
    warp_itk.SetSpacing(fix_itk.GetSpacing())
    # warp_itk = sitk.Cast(warp_itk, sitk.sitkInt16)
    # print(warp_itk.GetSize())
    # print(save_path)
    sitk.WriteImage(warp_itk, save_path)
    return warp_np
    

if __name__ == "__main__":

    image_dir = "/ailab/user/chenlingzhi/data/brain_xh/tumor_nifti_LPI/LBL"
    save_dir = "/ailab/user/chenlingzhi/data/brain_xh/tumor_nifti_regi/LBL"
    # fixed = "T1.nii.gz"
    # moving_list = ["T1-c.nii.gz", "T2.nii.gz", "FLAIR.nii.gz", "DWI0.nii.gz"]
    fixed = "T1.nii.gz"
    moving_list = ["T1.nii.gz", "T2.nii.gz", "T1-c.nii.gz", "FLAIR.nii.gz", "DWI0.nii.gz"]

    print("Do registration...")
    cnt = 0
    for f in tqdm(os.listdir(image_dir)):
        fixed_path = os.path.join(save_dir, f, fixed)
        if not os.path.isfile(fixed_path):
            print("Fixed image not exist: %s" % fixed_path)
            continue
        
        cnt += 1
        
        for moving in moving_list:
            registration = moving
            if moving == fixed:
                continue
            moving_path = os.path.join(image_dir, f, moving)
            if not os.path.isfile(moving_path):
                print("Moving image not exist: %s" % moving_path)
                continue
            save_path = os.path.join(save_dir, f, registration)
            if os.path.isfile(save_path):
                continue
            try:
                # print("Processing %s" % moving)
                do_registration(fixed_path, moving_path, save_path)
            except:
                print(f)
            # print("processing %s:" % f)
    print(cnt)