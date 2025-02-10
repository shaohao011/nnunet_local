export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/data/single_modal/nnUNet_format_raw_data_base
python=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/myconda/conda_dir/envs/local_env/bin/python
work_space=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local
export nnUNet_preprocessed=${work_space}/preprocess-results/single_modal/nnunet_official

export nnUNet_n_proc_DA=24
export nnUNet_compile=0
# core params
exp_name=nnunet_official
model_type=nnunet_official
dataset_id=2

test_image_folder=$nnUNet_raw/Dataset00${dataset_id}_Task001/imagesTs
test_label_folder=$nnUNet_raw/Dataset00${dataset_id}_Task001/labelsTs
predict_folder=${work_space}/pred-results/single_modal/${exp_name}/Dataset00${dataset_id}_Task001
export nnUNet_results=${work_space}/exp-results/single_modal/${exp_name}
pretrain_path=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/DPLS/pretrain_weights/${model_type}.pth
# training
# ${python} local_run_training.py ${dataset_id} 2d 0 -num_gpus 1 --model_name ${model_type} --resnet_pre_ckpt ${pretrain_path}

# inference
# ${python} local_nnunet_inference.py -i ${test_image_folder} -o ${predict_folder} -d ${dataset_id} -c 2d -chk checkpoint_best.pth -f 0 --model_name ${model_type} --resnet_pre_ckpt ${pretrain_path}

# cal metrics
# ${python} cal_metric.py --pred_folder ${predict_folder} --label_folder ${test_label_folder} > ${predict_folder}/metric.txt

# visualize
${python} local_visualize.py --pred_folder ${predict_folder} --label_folder ${test_label_folder} --dataset_id ${dataset_id}
