from typing import Union,List,Tuple
from torch import nn
import segmentation_models_pytorch as smp
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


@staticmethod
def build_network_architecture(
                                model_name,
                                resnet_pre_ckpt,
                                architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
    assert model_name!=''
    if 'mit' in model_name:
        print("[!] using custom model: ",model_name)
        model = smp.Unet(
        encoder_name=model_name, 
        encoder_weights='imagenet', 
        classes=num_output_channels, 
        activation=None,
        model_path = resnet_pre_ckpt)
        if model_name.split('_')[-1]=='b0':
            model.encoder.patch_embed1.proj = nn.Conv2d(num_input_channels,32,kernel_size=(7,7),stride=(4,4),padding=(3,3))
        elif model_name.split('_')[-1] in ['b3','b4']:
            model.encoder.patch_embed1.proj = nn.Conv2d(num_input_channels,64,kernel_size=(7,7),stride=(4,4),padding=(3,3))
    elif 'nnunet_official' in model_name:   
        print("[!] using nnunet model: ",model_name)
        model = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
    else:
        raise NotImplementedError
    return model