    
import numpy as np
from PIL import Image   
import SimpleITK as sitk
import ants
import os
import pydicom
import warnings


# 忽略所有警告
warnings.filterwarnings("ignore")

def array2nifti():
    img_slice = Image.fromarray(slice_data_normalized)
    png_filename =(f'slice_test.png')
    img_slice.save(png_filename)
    rgb_label_image = np.zeros((336, 336, 3), dtype=np.uint8)
    # 应用颜色映射
    color_map = {
     't2w':{
        1: (0, 0, 255),    # 蓝色
        2: (255, 0, 255),  # 粉色
        3: (255, 128, 0)   # 橙色 IMH
        }}
    for label, color in color_map.items():
        mask = (mask_data == label)
        rgb_label_image[mask] = color
        
        mask_label = (nii_label_img == label)
        rgb_label_image[mask_label] = color
        
def load_bmp_folder(dir):
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixels = np.array(image)
    class_map = np.zeros(pixels.shape[:2], dtype=int)
    for color, value in color_mapping.items():
        class_map = np.where(np.all(pixels == color, axis=-1), value, class_map)
    data = class_map

def check_real_num(num):
    if num.isdigit():
        num = int(num)
        return num
    else:
        return None
 
def regis_from_arrays():
    mode_name = 'ElasticSyN'
    fix_np = np.random.randn(336,336,7).astype('float32')
    mov_np = np.random.randn(256,256,3).astype('float32')
    print(mov_np.shape)
    # fix_np = sitk.GetArrayFromImage(fix_itk).astype('float32')
    # mov_np = sitk.GetArrayFromImage(mov_itk).astype('float32') # pure numpy
    out = ants.registration(ants.from_numpy(fix_np), 
                    ants.from_numpy(mov_np), type_of_transform=mode_name)#'Syn
    print(out['warpedmovout'].numpy().shape)
    
def load_onebyone_dicom_series(directory):
    image_list = []
    dcm_files = [i for i in os.listdir(directory) if i.endswith('.dcm')]
    # print(sorted(dcm_files))
    for file_path in sorted(dcm_files):
        ds = pydicom.dcmread(os.path.join(directory,file_path),force=True)
        data = ds.pixel_array.astype('float32')
        image_list.append(data)
    image_data = np.stack(image_list,axis=-1).transpose(2,0,1).astype('float32')
    # print(image_data.max(),"[!]",np.sum(image_data))
    image = sitk.GetImageFromArray(image_data)
    return image
    
def load_allofones_dicom_series(directory,return_com=False,mode_name=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    # print(dicom_names)
    # 设置文件名列表
    reader.SetFileNames(dicom_names)
    # 读取图像
    image = reader.Execute()
    # 设置图像的方向为 RAS 编码
    # image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # RAS 方向矩阵
    # print(image.GetSize()) # 
    # print(image.GetSpacing()) # (1.171875, 1.171875, 22.000001267099023)
    # print(image.GetOrigin()) # (94.929587979801, -131.97726628082, 236.855159932515)原点
    # print(image.GetDirection()) 
    # fix_np = sitk.GetArrayFromImage(image).astype('float32')
    if return_com:
        com_flag = len(dicom_names)==len(os.listdir(directory))/2 if mode_name in ["T2W","PSIR"] else len(dicom_names)==len(os.listdir(directory))
        # print(len(dicom_names),len(os.listdir(directory)))
        return dicom_names,com_flag
    return image    
    
# def regis_slice_level():
#     base_folder = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/total_200/four/20180308 wuhaiyuan"
#     modal_folder = os.path.join(base_folder,'T2W')

# def regis_modal_level():
#     pass

def regis_std(fix_itk,mov_itk,mode_name="ElasticSyN"):
    fix_np = sitk.GetArrayFromImage(fix_itk).astype('float32')
    mov_np = sitk.GetArrayFromImage(mov_itk).astype('float32') # pure numpy
    # print(fix_np.shape,np.sum(fix_np),mov_np.shape,np.sum(mov_np))
    if fix_np.shape!=mov_np.shape:
        # print(fix_np.shape,mov_np.shape,"scaling....")
        # 使用 SimpleITK 进行插值
        mov_itk_resampled = sitk.Resample(mov_itk, fix_itk, sitk.Transform(), 
                                          sitk.sitkLinear, 0.0, mov_itk.GetPixelID())
        mov_np = sitk.GetArrayFromImage(mov_itk_resampled).astype('float32')
        print("Resampled mov_np shape:", mov_np.shape)
        # exit()
    
    out = ants.registration(ants.from_numpy(fix_np), 
                    ants.from_numpy(mov_np), type_of_transform=mode_name)#'Syn
    # disp_np = ants.image_read(out['fwdtransforms'][0]).numpy() # 这是获得位移场
    # print(fix_np.shape, mov_np.shape)
    # print(out)
    warp_np = out['warpedmovout'].numpy() # 这是变形后的图像
    # print(warp_np.shape,np.sum(warp_np))
    if np.sum(warp_np)==0:
        print("wrong registration",np.sum(fix_np),np.sum(mov_np))
        # exit()
    warp_itk = sitk.GetImageFromArray(warp_np)
    warp_itk.SetDirection(fix_itk.GetDirection())
    warp_itk.SetOrigin(fix_itk.GetOrigin())
    warp_itk.SetSpacing(fix_itk.GetSpacing()) # origin orient [expect RAS]
    
    return warp_itk

def print_itk_info(temp_itk,name=''):
    print(name,'size',temp_itk.GetSize()) # 
    print(name,'spacing',temp_itk.GetSpacing()) # (1.171875, 1.171875, 22.000001267099023) # defaul (1 1 1)
    print(name,'origin',temp_itk.GetOrigin()) # (94.929587979801, -131.97726628082, 236.855159932515)原点 default 000
    print(name,'direction',temp_itk.GetDirection()) # RAS default

# regis_from_arrays()
# print(check_real_num('1_i'))
def test_regis():
    src_modal = 'PSIR'
    tgt_modals = ["T2W", "ECV", "eT1M", "nT1m", "PSIR", "T2-star", "T2m"]#[0:1]
    tgt_modals.remove(src_modal)
    for tgt_modal in tgt_modals:
        print(tgt_modal,"to",src_modal)
        fix_dir = f"/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/total_200_new/five_20190926lizhihua/{src_modal}"
        mov_dir = fix_dir.replace(src_modal,tgt_modal)
        load_func = load_allofones_dicom_series
        # load_func = load_onebyone_dicom_series
        fix_itk = load_func(fix_dir)
        print_itk_info(fix_itk,src_modal)
        sitk.WriteImage(fix_itk, f"{src_modal}.nii.gz")
        mov_itk = load_func(mov_dir)
        print_itk_info(mov_itk,tgt_modal)
        
        sitk.WriteImage(mov_itk, f"{tgt_modal}.nii.gz")
        
        mov_np = sitk.GetArrayFromImage(mov_itk).astype('float32') # pure numpy
        if np.sum(mov_np)==0:
            print(mov_dir)
            exit()
            
        warp_itk = regis_std(fix_itk,mov_itk)
        sitk.WriteImage(warp_itk, f"{src_modal}_{tgt_modal}.nii.gz")
       
    # fix_np = sitk.GetArrayFromImage(image).astype('float32')
    
    
def check_data_intergrety(base_dir):
    import shutil
    tgt_modals = ["T2W", "ECV", "eT1M", "nT1m", "PSIR", "T2-star", "T2m"]
    # tgt_modals.remove('T2W')
    # tgt_modals.remove('PSIR')
    count = 0
    for tgt_modal in tgt_modals:
        sample_num = 0
        for patient in os.listdir(base_dir):
            patient_path = os.path.join(base_dir,patient)
            if len(os.listdir(patient_path)) != 7:
                print(f"Warning: Directory {patient_path} does not have exactly 7 subdirectories. Deleting...")
                shutil.rmtree(patient_path)
                print(f"Directory {patient_path} has been deleted.")
                continue
            sample_num+=1
            # assert len(os.listdir(patient_path))==7,f"{patient_path}"
            modal_path = os.path.join(patient_path,tgt_modal)
            dicom_names,flag = load_allofones_dicom_series(modal_path,return_com=True,mode_name=tgt_modal)
            if not flag:
                print(modal_path,dicom_names)
                for dicom_name in [i for i in os.listdir(modal_path) if i.endswith('.dcm')]:
                    dicom_name = os.path.join(modal_path,dicom_name)
                    if dicom_name not in dicom_names:
                        os.remove(dicom_name)
                        if tgt_modal in ['T2W','PSIR']:
                            os.remove(dicom_name.replace('.dcm','.bmp')) 
                        print("delete ",dicom_name)
                # exit()
                count+=1
    if count>0:
        print("data is not complete")
    print('verify finished! sample num: ',sample_num)
        
# test_regis()
# check_data_intergrety()
# dir_temp = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/total_200_new/four_20200402guanjingen/PSIR"
# image = load_allofones_dicom_series(dir_temp)
# print_itk_info(image)

import os
import nibabel as nib
import numpy as np

def modify_and_save_nii_gz_1(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 .nii.gz 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取 NIFTI 文件
            nii_img = nib.load(input_path)
            data = nii_img.get_fdata()

            # 检查数据的形状
            if data.shape == (336, 336):
                # 修改形状为 (336, 336, 1)
                data = np.expand_dims(data, axis=-1)

                # 创建新的 NIFTI 图像
                new_nii_img = nib.Nifti1Image(data, nii_img.affine, nii_img.header)

                # 保存修改后的 NIFTI 文件
                nib.save(new_nii_img, output_path)
                print(f"Saved modified file: {output_path}")
            else:
                print(f"Skipping file {input_path} with shape {data.shape}")

# # 示例用法
# input_folder = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/Dataset001_Task001/labelsTs'
# output_folder = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/Dataset001_Task001/labelsTs_new"
# modify_and_save_nii_gz(input_folder, output_folder)

import os
import nibabel as nib
import numpy as np

def modify_and_save_nii_gz(input_folder):
    # 遍历输入文件夹中的所有 .nii.gz 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, filename)

            # 读取 NIFTI 文件
            nii_img = nib.load(input_path)
            data = nii_img.get_fdata()

            # 检查数据的形状
            if data.shape == (336, 336,1):
                # 将数据转换为整数
                data = data.astype(np.int32)

                # 修改形状为 (336, 336, 1)
                # data = np.expand_dims(data, axis=-1)

                # 创建新的 NIFTI 图像
                new_nii_img = nib.Nifti1Image(data, nii_img.affine, nii_img.header)

                # 保存修改后的 NIFTI 文件，覆盖原始文件
                nib.save(new_nii_img, input_path)
                print(f"Modified and saved file: {input_path}")
            else:
                print(f"Skipping file {input_path} with shape {data.shape}")

# # 示例用法
# input_folder = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/Dataset001_Task001/labelsTr'
# modify_and_save_nii_gz(input_folder)

import os
import re
def delete_non_matching_files(directory, target_extension):
    target_extension = target_extension.lower()

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_extension = os.path.splitext(file)[1].lower()
        if  file.endswith(target_extension):

            print(f"Deleting file: {file_path}")
            os.remove(file_path)
        # else:
        #     match = re.match(r'(.*?)(\d{4})\.nii\.gz', file)
        #     if match:
        #         prefix = match.group(1)
        #         sequence = match.group(2)
        #         if sequence != '0000':
        #             new_sequence = f'{int(sequence) - 1:04d}'
        #             new_file = f'{prefix}{new_sequence}.nii.gz'
        #             old_file_path = os.path.join(directory, file)
        #             new_file_path = os.path.join(directory, new_file)

        #             if os.path.exists(new_file_path):
        #                 print(f"Warning: {new_file_path} already exists. Skipping rename.")
        #             else:
        #                 print(f"Renaming: {old_file_path} -> {new_file_path}")
        #                 os.rename(old_file_path, new_file_path)
        #     else:
        #         print(f"Warning: File {file} does not match the expected pattern.")


# if __name__ == "__main__":
directory_to_clean = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/modal_6/Dataset001_Task001/imagesTs"  # 替换为你要清理的目录路径
target_extension = "_0001.nii.gz"  # 目标文件的扩展名

delete_non_matching_files(directory_to_clean, target_extension)
import re

def decrement_nifti_sequence(directory):
    """
    递归将目录下所有 NIFTI 文件的序号减一，但保留 0000.nii.gz 文件不变。

    :param directory: 要处理的目录路径
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                # 使用正则表达式提取文件名中的序号
                match = re.match(r'(.*?)(\d{4})\.nii\.gz', file)
                if match:
                    prefix = match.group(1)
                    sequence = match.group(2)
                    if sequence != '0000':
                        new_sequence = f'{int(sequence) - 1:04d}'
                        new_file = f'{prefix}{new_sequence}.nii.gz'
                        old_file_path = os.path.join(root, file)
                        new_file_path = os.path.join(root, new_file)
                        print(f"Renaming: {old_file_path} -> {new_file_path}")
                        os.rename(old_file_path, new_file_path)

# directory_to_clean = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/modal_6/Dataset001_Task001/imagesTr"  # 替换为你要清理的目录路径
# decrement_nifti_sequence(directory_to_clean)