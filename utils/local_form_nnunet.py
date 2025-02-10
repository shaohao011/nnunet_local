import json
import os
from PIL import Image
import numpy as np
import nibabel as nib
import SimpleITK as sitk

modal_name='T2W'
ref_json_path = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/DPLS/jsons/dataset_t2w_origin_mmodal.json'
tgt_name = "Old_SyN_registrated_200_T2W"
total_200_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/total_200"
base_folder = f"/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/{tgt_name}"
# valid_path = ""
dataset_id = 1
regis_name = "Old_SyN_registrated_200_T2W"
base_name = f"/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/multi_modal/{regis_name}"
out_nnunet_folder = os.path.join(base_name,f'Dataset00{dataset_id}_Task001')

modal_list =  ["T2W","PSIR", "ECV", "eT1m", "nT1m", "R2-star","T2m"] #NOTE important 
# modal_list =  ["PSIR","T2W","ECV", "eT1m", "nT1m", "R2-star","T2m"] #NOTE important 
# /ruishaohao-240108100042/renji/DPLS/data/processed_200_origin/one_20190827 qiumin_T2W_2.npy
total_data = json.load(open(ref_json_path))
training_data = total_data['training']
validation_data = total_data['validation'] # need to confirm
train_val = training_data+validation_data
test_data = total_data['test']


for idx,sample in enumerate(train_val):
    print("idx: ",idx)
    # if idx<=63:
    #     continue
    case_name = sample['image'].split('/')[-1].split('_')[:-2]
    # temp_list = [i.replace(' ','_') for i in case_name]
    temp_list = case_name
    temp_list.insert(1,'/')
    case_name = ''.join(temp_list) # one/xxx/ #T2W
    if not os.path.isdir(os.path.join(base_folder,case_name)):
        print("missing folder...",os.path.join(base_folder,case_name))
        idx -= 1
        continue
    modals_folder = [os.path.join(base_folder,case_name,modal) for modal in modal_list]
    # print(case_name)
    # len_slices = len(os.listdir(modals_folder[0]+"/slices"))
    slice_id = int(sample['image'].split('/')[-1].split('_')[-1].replace('.npy',''))-1 # 从0开始计数
    # for i in range(len_slices):
    modals_slices = [os.path.join(modal_folder,'compose.nii.gz') for modal_folder in modals_folder]
    # .nii图像和原始dcm存在一定的不同
    compose_list = [nib.load(modal_slices).get_fdata().transpose(1,0,2) for modal_slices in modals_slices] # 需要和其它模态合并
    print(compose_list[0].shape,slice_id,case_name)
    cur_slice_compose = np.stack([data[:,:,slice_id] for data in compose_list],axis=-1)
    assert cur_slice_compose.shape[-1]==7
    BMP_folder = os.path.join(total_200_path,case_name,f"{modal_name}_seg")
    Modal_folder = os.path.join(total_200_path,case_name,f"{modal_name}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(Modal_folder)
    print(dicom_names)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    exit()
    mask_data = np.load(sample['mask'])[..., np.newaxis]  # 需要追根溯源去找合适的mask
    
    # print(cur_slice_compose.shape,mask_data.shape,np.unique(mask_data))
    # 获取病例的唯一标识符
    case_id = idx
    
    # 创建输出文件夹
    output_folder = out_nnunet_folder
    os.makedirs(os.path.join(output_folder,"imagesTr"),exist_ok=True)
    os.makedirs(os.path.join(output_folder,"labelsTr"),exist_ok=True)
    for j in range(len(modal_list)):
        nii_array = cur_slice_compose[:,:,j:j+1]
        output_filename = f'case_{case_id:04d}_{j:04d}.nii.gz'
        output_path = os.path.join(output_folder,"imagesTr",output_filename)
        nii_img = nib.Nifti1Image(nii_array, np.eye(4))
        nib.save(nii_img, output_path)

    # 存储标签
    mask_filename = f'case_{case_id:04d}.nii.gz'
    mask_path = os.path.join(output_folder,"labelsTr", mask_filename)
    
    # 将标签转换为 NIfTI 格式
    nii_mask = nib.Nifti1Image(mask_data.astype(np.int8), np.eye(4))
    nib.save(nii_mask, mask_path)
    
for idx,sample in enumerate(test_data):
    print("idx: ",idx)
    case_name = sample['image'].split('/')[-1].split('_')[:-2]
    # temp_list = [i.replace(' ','_') for i in case_name]
    temp_list = case_name
    temp_list.insert(1,'/')
    case_name = ''.join(temp_list) # one/xxx/ #T2W
    if not os.path.isdir(os.path.join(base_folder,case_name)):
        print("missing folder...",os.path.join(base_folder,case_name))
        idx -= 1
        continue
    modals_folder = [os.path.join(base_folder,case_name,modal) for modal in modal_list]
    # print(case_name)
    # len_slices = len(os.listdir(modals_folder[0]+"/slices"))
    slice_id = int(sample['image'].split('/')[-1].split('_')[-1].replace('.npy',''))-1 # 从0开始计数
    # for i in range(len_slices):
    modals_slices = [os.path.join(modal_folder,'compose.nii.gz') for modal_folder in modals_folder]
    # .nii图像和原始dcm存在一定的不同
    compose_list = [nib.load(modal_slices).get_fdata().transpose(1,0,2) for modal_slices in modals_slices] # 需要和其它模态合并
    cur_slice_compose = np.stack([data[:,:,slice_id] for data in compose_list],axis=-1)
    assert cur_slice_compose.shape[-1]==7
    mask_data = np.load(sample['mask'])[..., np.newaxis]
    
    # print(cur_slice_compose.shape,mask_data.shape,np.unique(mask_data))
    # 获取病例的唯一标识符
    case_id = idx
    
    # 创建输出文件夹
    output_folder = out_nnunet_folder
    os.makedirs(os.path.join(output_folder,"imagesTs"),exist_ok=True)
    os.makedirs(os.path.join(output_folder,"labelsTs"),exist_ok=True)
    for j in range(len(modal_list)):
        nii_array = cur_slice_compose[:,:,j:j+1]
        output_filename = f'case_{case_id:04d}_{j:04d}.nii.gz'
        output_path = os.path.join(output_folder,"imagesTs",output_filename)
        nii_img = nib.Nifti1Image(nii_array, np.eye(4))
        nib.save(nii_img, output_path)

    # 存储标签
    mask_filename = f'case_{case_id:04d}.nii.gz'
    mask_path = os.path.join(output_folder,"labelsTs", mask_filename)
    
    # 将标签转换为 NIfTI 格式
    nii_mask = nib.Nifti1Image(mask_data.astype(np.int8), np.eye(4))
    nib.save(nii_mask, mask_path)
    exit()
        