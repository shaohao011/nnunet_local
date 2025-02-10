import os
import json
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from utils.useful_tools import *
from PIL import Image
import shutil
def append_slice(data_list,output_dir):
    pass



# 假设配准后的图像已经生成，这里我们直接使用这些图像
# 定义模态名称和对应的编码
modal_dict = {'T2W': 0,'PSIR': 1,'ECV': 2,"eT1M":3, "nT1m":4, "T2-star":5, "T2m":6}

# 定义数据目录和保存目录
data_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/total_200_new'
output_dir = 'out_temp'
os.makedirs(output_dir, exist_ok=True)

# 定义数据集划分比例
train_ratio = 0.33
val_ratio = 0.33
test_ratio = 0.33

# 定义随机种子
random_seed = 42

# 遍历每个patient目录
patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# 用于保存数据集划分的信息
dataset_split = {
    'train': [],
    'val': [],
    'test': []
}

# 用于保存case级别的信息
case_info = []
tgt_modal = 'T2W'
other_modals = ["T2W", "ECV", "eT1M", "nT1m", "PSIR", "T2-star", "T2m"]
other_modals.remove(tgt_modal)
# 定义颜色映射
color_mapping_t2w = {
(255, 0, 0):0, # 红色 error when labeling  
(0, 0, 255):1, #蓝色  正常
(255, 0, 255):2, #粉色 水肿
(255, 128, 0):3, # 橙色 IMH
(239, 239, 239):3, # 浅灰色 error when labeling
}
color_mapping_psir = {
(0, 0, 255):1, #蓝色
(255, 0, 0):2, #红色 # 梗死心肌
(255, 0, 255):2, #粉色 error when labeling
(0, 255, 0):3 #绿色 # MVO
}

count = 0
for index_total,patient_dir in enumerate(sorted(patient_dirs)):
    print("index: ",index_total)
    # if index_total == 3:
    #     break
    # case_info[patient_dir] = {}
    patient_data = {}
    patient_data[patient_dir] = []
    patient_path = os.path.join(data_dir, patient_dir)
    modal_dirs = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
    assert len(modal_dirs)==7, f"{patient_path},{modal_dirs}"
    # 获取T2W模态的图像
    fix_dir = os.path.join(patient_path, tgt_modal)
    fix_itk = load_allofones_dicom_series(fix_dir)
    fix_np = sitk.GetArrayFromImage(fix_itk).astype('float32')
    # print(fix_np.shape) # 9 336 336
    num_slices = fix_np.shape[0]
    
    # 提前配准
    mov_np_dict = {}
    for modal_name in other_modals:
        mov_dir = os.path.join(patient_path, modal_name)
        mov_itk = load_allofones_dicom_series(mov_dir)
        regis_mov_itk = regis_std(fix_itk,mov_itk,mode_name="ElasticSyN")
        mov_np = sitk.GetArrayFromImage(regis_mov_itk).astype('float32')
        mov_np_dict[modal_name] = mov_np
    
    # 保存每个模态的切片
        # print(mov_np.shape) # 9 336 336
    for slice_id in range(num_slices):
        #######################save labels########################
        slice_label_data = Image.open(sorted([os.path.join(fix_dir,i) for i in os.listdir(fix_dir) if i.endswith(".bmp")])[slice_id])
        if slice_label_data.mode != 'RGB':
            slice_label_data = slice_label_data.convert('RGB')
        pixels = np.array(slice_label_data)
        if "PSIR" == tgt_modal:
            color_mapping = color_mapping_psir
        elif "T2W" == tgt_modal:
            color_mapping = color_mapping_t2w
        class_map = np.zeros(pixels.shape[:2], dtype=int)
        for color, value in color_mapping.items():
            class_map = np.where(np.all(pixels == color, axis=-1), value, class_map)
        output_label_file = os.path.join(output_dir,"labels", f'case_{count:04d}.nii.gz')
        os.makedirs(os.path.join(output_dir,"labels"),exist_ok=True)
        slice_label_img = nib.Nifti1Image(class_map.astype(np.int8), affine=np.eye(4))
        nib.save(slice_label_img, output_label_file)
        
        for modal_name in modal_dirs:
            if modal_name == tgt_modal:
                mov_np = fix_np
            else:
                mov_np = mov_np_dict[modal_name]
                    
            slice_data = mov_np[slice_id,:, : ][:,:,np.newaxis]
            output_file = os.path.join(output_dir,"images", f'case_{count:04d}_{modal_dict[modal_name]:04d}.nii.gz')
            os.makedirs(os.path.join(output_dir,"images"),exist_ok=True)
            slice_img = nib.Nifti1Image(slice_data, affine=np.eye(4))
            nib.save(slice_img, output_file)
            patient_data[patient_dir].append(output_file)
            
            # case_info[patient_dir].append(output_file)
            # case_info.append({patient_dir:output_file})
            # case_info.append({
            #     'patient_id': patient_dir,
            #     'slice_id': slice_id,
            #     'modal_name': modal_name,
            #     'file_path': output_file
            # })
        count+=1
    case_info.append(patient_data)
            

# 划分数据集
train_cases, temp_cases = train_test_split(case_info, test_size=(val_ratio + test_ratio), random_state=random_seed)
val_cases, test_cases = train_test_split(temp_cases, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

dataset_split['train'] = train_cases
dataset_split['val'] = val_cases
dataset_split['test'] = test_cases

nnunet_folder = "nnunet_multi_modal/Dataset001_Task001"
os.makedirs(os.path.join(nnunet_folder,"imagesTr"),exist_ok=True)
os.makedirs(os.path.join(nnunet_folder,"imagesTs"),exist_ok=True)
os.makedirs(os.path.join(nnunet_folder,"labelsTr"),exist_ok=True)
os.makedirs(os.path.join(nnunet_folder,"labelsTs"),exist_ok=True)
train_val = train_cases+val_cases
for sample in train_val:
    for patient_name, slice_lists in sample.items():
        for slice in slice_lists:
            shutil.copy(slice,slice.replace(f'{output_dir}/images',f"{nnunet_folder}/imagesTr"))
            # print(os.path.splitext(slice))
            mask_name = os.path.splitext(slice)[0][:-9]+".nii.gz"
            # mask_folder = '/'.join(slice.replace('images','labels').split('/')[:-1]).lstrip('/')
            mask_name = mask_name.replace('images','labels')
            print(mask_name)
            shutil.copy(mask_name,mask_name.replace(f'{output_dir}/labels',f"{nnunet_folder}/labelsTr"))
        
for sample in test_cases:
    for patient_name, slice_lists in sample.items():
        for slice in slice_lists:
            shutil.copy(slice,slice.replace(f'{output_dir}/images',f"{nnunet_folder}/imagesTs"))
            mask_name = os.path.splitext(slice)[0][:-9]+".nii.gz"
            # mask_folder = '/'.join(slice.replace('images','labels').split('/')[:-1]).lstrip('/')
            mask_name = mask_name.replace('images','labels')
            print(mask_name)
            shutil.copy(mask_name,mask_name.replace(f'{output_dir}/labels',f"{nnunet_folder}/labelsTs"))



# 保存数据集划分信息到json文件
with open('dataset_split.json', 'w') as f:
    json.dump(dataset_split, f, indent=4)

print("数据集划分完成，并保存为 dataset_split.json")