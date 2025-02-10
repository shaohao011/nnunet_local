import os
import shutil
import re

# 定义标准模态名称
standard_modalities = ["T2W", "ECV", "eT1M", "nT1m", "PSIR", "T2-star", "T2m"]

# 定义原始目录和目标目录
source_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/origin_total_200/total_200'
target_dir = 'total_200_new'

# 创建目标目录
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历one, two, three目录
for subdir in sorted(os.listdir(source_dir)):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        # 遍历病人目录
        for patient_dir in os.listdir(subdir_path):
            patient_dir_path = os.path.join(subdir_path, patient_dir)
            if os.path.isdir(patient_dir_path):
                # 删除病人名称中的空格
                patient_name = patient_dir.replace(' ', '')
                new_patient_dir = f"{subdir}_{patient_name}"
                new_patient_dir_path = os.path.join(target_dir, new_patient_dir)
                
                # 创建新的病人目录
                # if "20231121chenyumei"  in new_patient_dir_path:
                #     continue
                if not os.path.exists(new_patient_dir_path):
                    os.makedirs(new_patient_dir_path)
                
                # 遍历模态目录
                for modality_dir in os.listdir(patient_dir_path):
                    modality_dir_path = os.path.join(patient_dir_path, modality_dir)
                    if os.path.isdir(modality_dir_path):
                        # 检查模态名称是否在标准列表中
                        if modality_dir in standard_modalities:
                            new_modality_dir_path = os.path.join(new_patient_dir_path, modality_dir)
                            
                            # 获取所有.dcm文件并排序
                            dcm_files = sorted([f for f in os.listdir(modality_dir_path) if f.endswith('.dcm')]) # DICOM是按顺序存储的，那么只需要用它做索引
                            if len(dcm_files)==0:print(modality_dir_path)
                            bmp_files = sorted([f for f in os.listdir(modality_dir_path) if f.endswith('.bmp') and os.path.splitext(f)[0].isdigit()])
                            pass_flag = len(dcm_files) == len(bmp_files) or (len(dcm_files)!=0 and len(bmp_files)==0)
                            if len(dcm_files)==0:
                                print("Error",modality_dir_path)
                                continue
                            if not pass_flag:
                                print(f"##########,{dcm_files},{bmp_files},{modality_dir_path}")
                                continue
                            if not os.path.exists(new_modality_dir_path):
                                os.makedirs(new_modality_dir_path)
                            # 遍历.dcm文件
                            for temp_i, dcm_file in enumerate(dcm_files):
                                dcm_file_path = os.path.join(modality_dir_path, dcm_file)
                                dcm_name = os.path.splitext(dcm_file)[0]
                                dcm_number = temp_i+1 # 新的文件名称
                                
                                # 提取数字部分并转换为单个数字
                                # try:
                                #     dcm_number = re.match(r'\d+', dcm_name).group(0)
                                #     if len(dcm_number) > 1:
                                #         dcm_number = dcm_number[0]
                                #         assert dcm_number!=0
                                # except:
                                #     if "IMG" in dcm_name or "img00" in dcm_name:
                                #         # print(dcm_name,dcm_name.split('-')[-1],re.match(r'\d+',dcm_name.split('-')[-1]))
                                #         # dcm_number = re.match(r'\d+',dcm_name.split('-')[-1]).group(0)
                                #         # dcm_number = int(dcm_number) % 10
                                #         dcm_number = temp_i+1
                                #         assert dcm_number!=0
                                #         # print(dcm_number,dcm_name,dcm_file_path)
                                #         # exit()
                                #     else:
                                #         print(dcm_number,dcm_name,dcm_file_path,"fistt")
                                #         exit()
                               
                                
                                new_dcm_file = f"{dcm_number}.dcm"
                                new_dcm_file_path = os.path.join(new_modality_dir_path, new_dcm_file)
                                shutil.copy(dcm_file_path, new_dcm_file_path)
                                
                                # 检查是否有对应的.bmp文件
                                # if patient_name=="20220726wujian":
                                #     print(dcm_name)
                                #     exit()
                                # for bmp_file in bmp_files:
                                    # bmp_name = os.path.splitext(bmp_file)[0]
                                    # if dcm_name in bmp_name and re.match(r'^\d+$', bmp_name):
                                if len(bmp_files)>0:
                                    bmp_file = bmp_files[temp_i]
                                    bmp_file_path = os.path.join(modality_dir_path, bmp_file)
                                    new_bmp_file_path = os.path.join(new_modality_dir_path, f"{dcm_number}.bmp")
                                    shutil.copy(bmp_file_path, new_bmp_file_path)
                                    # break
                        else:
                            # 检查是否是大小写问题
                            corrected_modality = None
                            for std_modality in standard_modalities:
                                if std_modality.lower() == modality_dir.lower():
                                    corrected_modality = std_modality
                                    break
                                elif modality_dir=="T2WI":
                                    corrected_modality = "T2W"
                                    break
                                elif modality_dir=="R2-star":
                                    corrected_modality = "T2-star"
                                    break
                            if corrected_modality:
                                new_modality_dir_path = os.path.join(new_patient_dir_path, corrected_modality)
                                    # 获取所有.dcm文件并排序
                                dcm_files = sorted([f for f in os.listdir(modality_dir_path) if f.endswith('.dcm')]) # DICOM是按顺序存储的，那么只需要用它做索引
                                if len(dcm_files)==0:print(modality_dir_path)
                                bmp_files = sorted([f for f in os.listdir(modality_dir_path) if f.endswith('.bmp') and os.path.splitext(f)[0].isdigit()])
                                pass_flag = len(dcm_files) == len(bmp_files) or (len(dcm_files)!=0 and len(bmp_files)==0)
                                if len(dcm_files)==0:
                                    print("Error",modality_dir_path)
                                    continue
                                if not pass_flag:
                                    print(f"##########,{dcm_files},{bmp_files},{modality_dir_path}")
                                    continue
                                if not os.path.exists(new_modality_dir_path):
                                    os.makedirs(new_modality_dir_path)
                                # 遍历.dcm文件
                                for temp_i, dcm_file in enumerate(dcm_files):
                                    dcm_file_path = os.path.join(modality_dir_path, dcm_file)
                                    dcm_name = os.path.splitext(dcm_file)[0]
                                    dcm_number = temp_i+1 # 新的文件名称
                                    
                                    # 提取数字部分并转换为单个数字
                                    # try:
                                    #     dcm_number = re.match(r'\d+', dcm_name).group(0)
                                    #     if len(dcm_number) > 1:
                                    #         dcm_number = dcm_number[0]
                                    #         assert dcm_number!=0
                                    # except:
                                    #     if "IMG" in dcm_name or "img00" in dcm_name:
                                    #         # print(dcm_name,dcm_name.split('-')[-1],re.match(r'\d+',dcm_name.split('-')[-1]))
                                    #         # dcm_number = re.match(r'\d+',dcm_name.split('-')[-1]).group(0)
                                    #         # dcm_number = int(dcm_number) % 10
                                    #         dcm_number = temp_i+1
                                    #         assert dcm_number!=0
                                    #         # print(dcm_number,dcm_name,dcm_file_path)
                                    #         # exit()
                                    #     else:
                                    #         print(dcm_number,dcm_name,dcm_file_path,"fistt")
                                    #         exit()
                                
                                    
                                    new_dcm_file = f"{dcm_number}.dcm"
                                    new_dcm_file_path = os.path.join(new_modality_dir_path, new_dcm_file)
                                    shutil.copy(dcm_file_path, new_dcm_file_path)
                                    
                                    # 检查是否有对应的.bmp文件
                                    # if patient_name=="20220726wujian":
                                    #     print(dcm_name)
                                    #     exit()
                                    # for bmp_file in bmp_files:
                                        # bmp_name = os.path.splitext(bmp_file)[0]
                                        # if dcm_name in bmp_name and re.match(r'^\d+$', bmp_name):
                                    if len(bmp_files)>0:
                                        bmp_file = bmp_files[temp_i]
                                        bmp_file_path = os.path.join(modality_dir_path, bmp_file)
                                        new_bmp_file_path = os.path.join(new_modality_dir_path, f"{dcm_number}.bmp")
                                        shutil.copy(bmp_file_path, new_bmp_file_path)
                                        # break
                            else:
                                print(f"Error: {modality_dir_path} in {patient_dir_path} is not a valid modality.")
                                exit()
                                
from utils.useful_tools import check_data_intergrety
print("Data cleaning and copying completed.")
check_data_intergrety(target_dir)