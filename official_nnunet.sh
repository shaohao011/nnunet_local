export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/nnunet_multi_modal/modal_6
export nnUNet_preprocessed=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/preprocess-results/multi_modal_modal_6
export nnUNet_n_proc_DA=32
export nnUNet_compile=0
export nnUNet_results=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/codes/nnunet_local/exp-results/multi_modal_modal_6
nnUNetv2_plan_and_preprocess -d 1 -gpu_memory_target 80

# python run_training.py 1 2d 0 -num_gpus 1
# python local_nnunet_inference.py -i ${test_image_path} -o ./T2W -d 1 -c 2d --save_probabilities -chk checkpoint_best.pth -f 0