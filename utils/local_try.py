import os
import SimpleITK as sitk
import ants
def remove_spaces_from_subdirs(main_directory):
    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            if ' ' in dir_name:
                new_name = dir_name.replace(' ', '_')  # 替换空格为下划线或其他字符
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f'Renamed: "{old_path}" to "{new_path}"')

# # 使用示例
# main_directory = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/QuickRigid_registrated_200_PSIR'  # 替换为你的主目录路径
# remove_spaces_from_subdirs(main_directory)

import os
import nibabel as nib
import numpy as np
from PIL import Image

def save_slices_as_png(nifti_file, output_dir):
    # 读取.nii.gz文件
    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = data.transpose((1, 0, 2))

    # print(data.shape)
    # exit()
    # 确保数据是三维的
    if data.ndim != 3:
        print(f"警告: 文件 {nifti_file} 不是三维数据。")
        return

    # 获取第三维度的切片数量
    num_slices = data.shape[2]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个切片并保存为.png
    for i in range(num_slices):
        slice_data = data[:, :, i]  # 取出第i个切片

        # 将切片数据归一化到0-255
        slice_data_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
        slice_data_normalized = slice_data_normalized.astype(np.uint8)

        # 保存为PNG文件
        img_slice = Image.fromarray(slice_data_normalized)
        png_filename = os.path.join(output_dir, f'slice_{i:03d}.png')
        img_slice.save(png_filename)

        print(f"保存切片 {png_filename}")

def process_directory(main_dir):
    # 遍历主目录及其所有子目录
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                nifti_file_path = os.path.join(root, file)
                output_dir = os.path.join(root, 'slices')  # 在同一目录下创建slices文件夹
                print(f"处理文件: {nifti_file_path}")
                save_slices_as_png(nifti_file_path, output_dir)



if __name__ == "__main__":
    main_directory = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/Old_SyN_registrated_200_T2W'  # 替换为你的主目录路径
    process_directory(main_directory)

