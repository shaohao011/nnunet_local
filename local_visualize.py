import json
import nibabel as nib
import numpy as np
from PIL import Image
import os
import os
import numpy as np
from monai.transforms import LoadImage
import argparse
from tqdm import tqdm

def create_soft_link(source, link_name):
    """
    创建一个软链接。

    :param source: 源文件或目录的路径
    :param link_name: 软链接的路径
    """
    try:
        os.symlink(source, link_name)
        # print(f"软链接已成功创建: {link_name} -> {source}")
    except FileExistsError:
        print(f"软链接已存在: {link_name}")
    except Exception as e:
        print(f"创建软链接时出错: {e}")

def overlay_and_save_bmp(nii_label_gray_img, rgb_pred_image, output_path,gray_path=None):
    # assert nii_label_gray_img.shape==rgb_pred_image.shape

    # 将 nii_label_gray_img 转换为三通道图像
    min_val = np.min(nii_label_gray_img)
    max_val = np.max(nii_label_gray_img)
    normalized_image = ((nii_label_gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    nii_label_gray_img_3d = np.stack([normalized_image]*3, axis=-1)
    # 将 rgb_pred_image 叠加到 nii_label_gray_img 上
    overlay_image = np.where(rgb_pred_image !=(0,0,0), rgb_pred_image, nii_label_gray_img_3d)

    # 将 numpy 数组转换为 PIL 图像
    overlay_image_pil = Image.fromarray(overlay_image)
    overlay_image_pil.save(output_path, format='BMP')
    nii_label_gray_img_save = Image.fromarray(nii_label_gray_img_3d)
    if gray_path:
        nii_label_gray_img_save.save(gray_path, format='BMP')
    return overlay_image_pil,nii_label_gray_img_save
def draw_on_bmp(args):
    pred_folder = args.pred_folder
    label_folder = args.label_folder
    modal_name = {1:'t2w',2:'psir'}[args.dataset_id]
    visualize_bmp_folder = args.pred_folder.replace('pred-results','visualize-results')
    os.makedirs(visualize_bmp_folder,exist_ok=True)

    json_path = os.path.join(args.base_json_folder,f"dataset_{modal_name}_origin.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    test_data = data['test']

    # 定义颜色映射
    color_map_dict = {
        't2w':{
        1: (0, 0, 255),    # 蓝色
        2: (255, 0, 255),  # 粉色
        3: (255, 128, 0)   # 橙色 IMH
        },
        'psir':
            {
        1: (0, 0, 255),    # 蓝色
        2: (255, 0, 0),    # 红色 # 梗死心肌
        3: (0, 255, 0)     # 绿色 # MVO
            }
    }
    
    color_map = color_map_dict[modal_name] 

    # 遍历 test 中的每个条目
    # for idx, entry in enumerate(test_data):
    for idx, label_file_name in enumerate(sorted(os.listdir(label_folder))):
        # 加载 .nii.gz 文件
        nii_pred_path = os.path.join(pred_folder, label_file_name)
        nii_label_path = os.path.join(label_folder, label_file_name)
        nii_label_gray_path = nii_label_path.replace('labelsTs','imagesTs').replace('.nii.gz','_0000.nii.gz')
        
        
        nii_pred_img = np.squeeze(nib.load(nii_pred_path).get_fdata()) # 从 (336, 336, 1) 调整为 (336, 336)
        nii_label_img = np.squeeze(nib.load(nii_label_path).get_fdata()) # 从 (336, 336, 1) 调整为 (336, 336)
        nii_label_gray_img = np.squeeze(nib.load(nii_label_gray_path).get_fdata()) # 从 (336, 336, 1) 调整为 (336, 336)
        
        # 初始化一个空的 RGB 图像
        # print(nii_data.shape)
        assert nii_label_img.shape==nii_pred_img.shape
        height, width = nii_pred_img.shape
        rgb_pred_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_label_image = np.zeros((height, width, 3), dtype=np.uint8)
        # 应用颜色映射
        for label, color in color_map.items():
            mask = (nii_pred_img == label)
            rgb_pred_image[mask] = color
            
            mask_label = (nii_label_img == label)
            rgb_label_image[mask_label] = color

        # 保存为 BMP 文件
        file_folder = test_data[idx]['image'].split('/')[-1].replace('.npy','').split('_')
        file_folder_compose = os.path.join(visualize_bmp_folder,os.path.join(*file_folder[:-1]))
        os.makedirs(file_folder_compose,exist_ok=True)

        bmp_pred_path = os.path.join(file_folder_compose, f'pred_{file_folder[-1]}.bmp')
        # Image.fromarray(rgb_pred_image).save(bmp_pred_path)
        
        bmp_label_path = os.path.join(file_folder_compose, f'gt_{file_folder[-1]}.bmp')
        # Image.fromarray(rgb_label_image).save(bmp_label_path)
        
        gray_path = os.path.join(file_folder_compose, f'image_{file_folder[-1]}.bmp')
        stack_pred,gray_image = overlay_and_save_bmp(nii_label_gray_img, rgb_pred_image, output_path=bmp_pred_path.replace('pred','stack_pred'),gray_path=gray_path)
        stack_gt,gray_image = overlay_and_save_bmp(nii_label_gray_img, rgb_label_image, output_path=bmp_label_path.replace('gt','stack_gt'))
        
        combined_width = gray_image.width + stack_gt.width + stack_pred.width
        combined_height = gray_image.height
        combined_image = Image.new('RGB', (combined_width, combined_height))
        # 将两张图像粘贴到新的图像中
        combined_image.paste(gray_image, (0, 0))
        combined_image.paste(stack_gt, (gray_image.width, 0))
        combined_image.paste(stack_pred, (gray_image.width*2, 0))

        # 保存为 BMP 文件
        combined_image.save(gray_path.replace('image','compose'), format='BMP')
        
         # copy .ii.gz file
        # tgt_gray_nil_path = os.path.join(file_folder_compose, f'image_{file_folder[-1]}.nii.gz')
        # create_soft_link(nii_label_gray_path,tgt_gray_nil_path)
        
        # tgt_pred_nil_path = os.path.join(file_folder_compose, f'pred_{file_folder[-1]}.nii.gz')
        # create_soft_link(nii_pred_path,tgt_pred_nil_path)
        
        # tgt_label_nil_path = os.path.join(file_folder_compose, f'gt_{file_folder[-1]}.nii.gz')
        # create_soft_link(nii_label_path,tgt_label_nil_path)
        

        print(f'Saved {bmp_pred_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_folder", default="",type=str)
    parser.add_argument("--label_folder", default="",type=str)
    parser.add_argument("--dataset_id", default=-1,type=int)
    parser.add_argument("--base_json_folder", default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/DPLS/jsons",type=str)
    
    args = parser.parse_args()
    draw_on_bmp(args)
   