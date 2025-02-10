import os
import json
import numpy as np
import nibabel as nib
import random 
from collections import defaultdict
import pydicom
import imageio
import SimpleITK as sitk
from tqdm import tqdm


def check_data_cross(training, validation, test):
    # 将每个集合的 ID 提取出来
    train_ids = {item["ID"] for item in training}
    val_ids = {item["ID"] for item in validation}
    test_ids = {item["ID"] for item in test}
    
    # 检查训练集和验证集是否有交集
    train_val_intersection = train_ids.intersection(val_ids)
    if train_val_intersection:
        print("训练集和验证集有重复数据！")
        print("重复的 ID:", train_val_intersection)
        print("重复的 item:")
        for item in training:
            if item["ID"] in train_val_intersection:
                print(item)
        for item in validation:
            if item["ID"] in train_val_intersection:
                print(item)
        raise ValueError
    
    # 检查训练集和测试集是否有交集
    train_test_intersection = train_ids.intersection(test_ids)
    if train_test_intersection:
        print("训练集和测试集有重复数据！")
        print("重复的 ID:", train_test_intersection)
        print("重复的 item:")
        for item in training:
            if item["ID"] in train_test_intersection:
                print(item)
        for item in test:
            if item["ID"] in train_test_intersection:
                print(item)
        raise ValueError
    
    # 检查验证集和测试集是否有交集
    val_test_intersection = val_ids.intersection(test_ids)
    if val_test_intersection:
        print("验证集和测试集有重复数据！")
        print("重复的 ID:", val_test_intersection)
        print("重复的 item:")
        for item in validation:
            if item["ID"] in val_test_intersection:
                print(item)
        for item in test:
            if item["ID"] in val_test_intersection:
                print(item)
        raise ValueError

    # 如果没有交集
    if not train_val_intersection and not train_test_intersection and not val_test_intersection:
        print("训练集、验证集和测试集之间没有数据交叉。")


def hirachy_sample(data_list):
    random.seed(220)  # 你可以选择任何整数作为种子
    grouped_data = defaultdict(list)
    for item in data_list:
        # print(item)
        grouped_data[str(int(item["mace"]))].append(item)
        
    training = []
    validation = []
    test = []
    for mace_value, items in grouped_data.items():
        random.shuffle(items)
        total = len(items)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        
        training.extend(items[:train_size])
        validation.extend(items[train_size:train_size + val_size])
        test.extend(items[train_size + val_size:])

    check_data_cross(training, validation, test)
    return training,validation,test
def get_num_str(input_str):
    try:
        data = int(input_str)
        return True
    except:
        return False
    

def get_slices_paths(data_item):
    cur_mod_dir = os.path.join(data_item['mod_parent'], "PSIR").replace('../data','data')
    if not os.path.exists(cur_mod_dir):
        print(data_item['mod_parent'])
        raise ValueError
    slice_files = sorted([os.path.join(cur_mod_dir, i) for i in os.listdir(cur_mod_dir) if (i not in [".DS_Store", 'mask']) and not (i.endswith('bmp')) and not (i.endswith('png'))])
    
    mask_files = sorted([os.path.join(cur_mod_dir, i) for i in os.listdir(cur_mod_dir) if (i not in [".DS_Store", 'img']) and (i.endswith('bmp')) and get_num_str(i[:-4])])
    
    slices = []
    slices_paths = []
    for slice_file in slice_files:
        if os.path.isdir(slice_file):
            inner_files = sorted([os.path.join(slice_file, i) for i in os.listdir(slice_file)])
            for inner_file in inner_files: 
                ds = pydicom.dcmread(inner_file, force=True)
                if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                pixel_array = ds.pixel_array
                slices.append(pixel_array)
                slices_paths.append(inner_file)
        else:
            ds = pydicom.dcmread(slice_file, force=True)
            if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            pixel_array = ds.pixel_array
            slices.append(pixel_array)
            slices_paths.append(slice_file)
                
    if len(slices_paths) == 0:
        print(cur_mod_dir)
        raise ValueError
    assert len(slices) == len(slices_paths)
    
    if len(slices) != len(mask_files):
        slices_paths = [i.replace('bmp','dcm') for i in mask_files]
        
    assert len(slices_paths) == len(mask_files),f"{mask_files}\n{slices_paths}"
    
    return slices, slices_paths,mask_files


# 颜色映射字典：将对应颜色映射为标签
color_mapping_psir = {
    (0, 0, 255): 1,     # 蓝色
    (255, 0, 0): 2,     # 红色 —— 梗死心肌
    (255, 0, 255): 2,   # 粉色 —— error when labeling
    (0, 255, 0): 3      # 绿色 —— MVO
}

# def convert_2d_dicom_to_nifti(dicom_path, output_path):
#     # 读取单个DICOM文件
#     reader = sitk.ImageFileReader()
#     reader.SetImageIO("GDCMImageIO")  # 使用GDCM库读取DICOM文件
#     reader.SetFileName(dicom_path)
    
#     # 执行读取操作
#     image = reader.Execute()
    
#     # 将图像保存为NIfTI格式
#     sitk.WriteImage(image, output_path)


def convert_2d_dicom_to_nifti(dicom_path, output_path):
    """
    将单个 2D DICOM 文件转换为 NIfTI 文件，并保留空间元信息。
    由于是2D图像，转换时在第三个维度上增加一个单层维度。

    为确保方向信息有效：
      1. 对 DICOM 中的行、列方向向量进行归一化，
      2. 计算正交的法向量（切片方向），
      3. 根据 DICOM 的 LPS 坐标系转换为 NIfTI 常用的 RAS 坐标系。
    """
    # 读取 DICOM 文件
    ds = pydicom.dcmread(dicom_path, force=True)
    if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # 获取像素数据（2D）
    img = ds.pixel_array  # 形状 (rows, cols)
    # 为符合 NIfTI 的 3D 格式，增加一个切片维度，变成 (rows, cols, 1)
    img_3d = img[..., np.newaxis]

    # 提取 DICOM 中的空间信息
    pixel_spacing = [float(x) for x in ds.PixelSpacing]  # [row_spacing, col_spacing]
    slice_thickness = 1.0  # 2D 图像时可设为1

    # 获取方向余弦向量（长度为6：前3为行方向，后3为列方向）
    orientation = np.array(ds.ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    # 对方向向量归一化（保证正交）
    row_cosine = row_cosine / np.linalg.norm(row_cosine)
    col_cosine = col_cosine / np.linalg.norm(col_cosine)
    # 计算切片法向量，并归一化
    normal = np.cross(row_cosine, col_cosine)
    normal = normal / np.linalg.norm(normal)

    # 图像原点（在 DICOM 中通常为左上角像素的坐标）
    origin = np.array(ds.ImagePositionPatient, dtype=np.float64)
    
    # 构造 DICOM 下的 affine（LPS 坐标系）
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * pixel_spacing[0]
    affine[:3, 1] = col_cosine * pixel_spacing[1]
    affine[:3, 2] = normal * slice_thickness
    affine[:3, 3] = origin

    # 将 affine 从 DICOM 的 LPS 坐标系转换为 NIfTI 的 RAS 坐标系
    LPS_to_RAS = np.diag([-1, -1, 1, 1])
    affine = LPS_to_RAS.dot(affine)

    # 可选：检查生成的方向轴
    axcodes = nib.orientations.aff2axcodes(affine)
    print("生成的图像方向轴：", axcodes)  # 例如 ('L','P','S') 或 ('R','A','S')

    # 构造并保存 NIfTI 图像
    nifti_img = nib.Nifti1Image(img_3d, affine)
    nifti_img.header.set_qform(affine, code=1)
    nifti_img.header.set_sform(affine, code=1)
    nifti_img.header['descrip'] = 'Converted from 2D DICOM slice (LPS->RAS conversion applied with normalized orientation)'
    nib.save(nifti_img, output_path)
    print(f"已保存影像 NIfTI 文件：{output_path}")

    # 返回 affine 以便后续 mask 转换使用
    return affine

def convert_2d_mask_to_nifti(mask_path, output_path, ref_affine):
    """
    将单个 2D segmentation mask（bmp 格式）转换为 NIfTI 文件，
    使用与影像相同的 affine 保持空间对应关系。

    对彩色 mask，根据 color_mapping_psir 将像素颜色转换为对应的标签：
      (0, 0, 255)   -> 1 (蓝色)
      (255, 0, 0)   -> 2 (红色, 梗死心肌)
      (255, 0, 255) -> 2 (粉色, error when labeling)
      (0, 255, 0)   -> 3 (绿色, MVO)
    """
    # 读取 mask 图像
    mask = imageio.imread(mask_path)
    
    if mask.ndim == 3:
        # 对于彩色图像，逐像素映射颜色至标签
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, label in color_mapping_psir.items():
            # 构造颜色数组（假定 mask 的通道顺序为 RGB）
            color_arr = np.array(color, dtype=mask.dtype)
            # 得到所有匹配该颜色的像素位置
            matches = np.all(mask == color_arr, axis=-1)
            label_mask[matches] = label
        mask = label_mask
    else:
        # 如果已经为灰度，则转换为 uint8 类型
        mask = mask.astype(np.uint8)
    
    # 为符合 NIfTI 格式，增加切片维度 (rows, cols, 1)
    mask_3d = mask[..., np.newaxis]
    nifti_mask = nib.Nifti1Image(mask_3d, ref_affine)
    nifti_mask.header.set_qform(ref_affine, code=1)
    nifti_mask.header.set_sform(ref_affine, code=1)
    nifti_mask.header['descrip'] = 'Converted from 2D segmentation mask with color mapping'
    nib.save(nifti_mask, output_path)
    print(f"已保存 mask NIfTI 文件：{output_path}")

    
# 假设你已经加载了JSON数据
# json_data = ...
with open("renji_full_label_dataset_1.json", 'r') as f:
    total_data = json.load(f)['total']

# 定义nnUNet的根目录
nnunet_base_dir = "nnUNet_raw"
dataset_name = "Dataset100_RenjiPSIR"

# 创建目标目录结构
target_dir = os.path.join(nnunet_base_dir, dataset_name)
os.makedirs(os.path.join(target_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labelsTr"), exist_ok=True)

os.makedirs(os.path.join(target_dir, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labelsTs"), exist_ok=True)

train_data,val_data,test_data = hirachy_sample(total_data)

test_data = val_data+test_data
print("training:", len(train_data))
print("test:", len(test_data))

train_item = []
global_idx = 0
for train_index, item in enumerate(train_data):
    # if train_index==5: break
    slices,slices_paths,mask_paths = get_slices_paths(item)
    # print(slices_paths)
    # print(mask_paths)
    item ['idxs'] = []
    print(item['mod_parent'])
    for image_path,mask_path in tqdm(zip(slices_paths,mask_paths)):
        image_filename = f"case_{global_idx:04d}_0000.nii.gz"
        mask_filename = f"case_{global_idx:04d}.nii.gz"
        output_file_name = os.path.join(nnunet_base_dir,dataset_name,"imagesTr",image_filename)
        output_seg_name = os.path.join(nnunet_base_dir,dataset_name,"labelsTr",mask_filename)
        if os.path.exists(output_file_name) and os.path.exists(output_seg_name):
            item ['idxs'].append(global_idx)
            global_idx+=1
            continue
        affine = convert_2d_dicom_to_nifti(image_path, output_file_name)
        convert_2d_mask_to_nifti(mask_path, output_seg_name, ref_affine=affine)
        
        item ['idxs'].append(global_idx)
        global_idx+=1
    train_item.append(item)
    
train_num = global_idx
test_item = []
for test_index, item in tqdm(enumerate(test_data)):
    # if test_index==5: break
    slices,slices_paths,mask_paths = get_slices_paths(item)
    # print(slices_paths)
    # print(mask_paths)
    item ['idxs'] = []
    print(item['mod_parent'])
    for image_path,mask_path in zip(slices_paths,mask_paths):
        image_filename = f"case_{global_idx:04d}_0000.nii.gz"
        mask_filename = f"case_{global_idx:04d}.nii.gz"
        output_file_name = os.path.join(nnunet_base_dir,dataset_name,"imagesTs",image_filename)
        output_seg_name = os.path.join(nnunet_base_dir,dataset_name,"labelsTs",mask_filename)
        
        if os.path.exists(output_file_name) and os.path.exists(output_seg_name):
            item ['idxs'].append(global_idx)
            global_idx+=1
            continue
        
        affine = convert_2d_dicom_to_nifti(image_path, output_file_name)
        convert_2d_mask_to_nifti(mask_path, output_seg_name, ref_affine=affine)
        item ['idxs'].append(global_idx)
        global_idx+=1
        
    test_item.append(item)
    
    
# 生成 dataset.json 文件
dataset_json = {
    "channel_names": {
        "0": "PSIR"
    },
    "labels": {
        "background": 0,
        "cls1": 1,
        "cls2": [2,3],
        "cls3": 3
    },
    "numTraining": train_num,
    # "numTest": len(json_data.get("test", [])),
    "file_ending": ".nii.gz",
    # "training": [
    #     {
    #         "image": os.path.join("imagesTr", f"case_{idx:04d}_0000.nii.gz"),
    #         "label": os.path.join("labelsTr", f"case_{idx:04d}.nii.gz")
    #     } for idx in range(len(train_data + val_data))
    # ],
    # "test": [
    #     {
    #         "image": os.path.join("imagesTs", f"case_{idx:04d}_0000.nii.gz"),
    #         "label": os.path.join("labelsTs", f"case_{idx:04d}.nii.gz")
    #     } for idx in range(len(json_data.get("test", [])))
    # ]
}

# 保存 dataset.json 文件
with open(os.path.join(target_dir, "dataset.json"), 'w') as f:
    json.dump(dataset_json, f, indent=4,ensure_ascii=False)

split_total = {"training":train_item,"test":test_item}


with open("renji_nnunet_slices_split.json",'w')as f:
    json.dump(split_total,f,indent=4,ensure_ascii=False)

print("NIFTI文件转换完成！")