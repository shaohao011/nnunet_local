    
    
from typing import Union,List,Tuple
from torch import nn
    
@staticmethod
def build_network_architecture(architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp
    model = smp.Unet(
    encoder_name='mit_b3', 
    encoder_weights='imagenet', 
    classes=3, 
    activation=None,
    model_path = f"/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/renji/DPLS/pretrain_weights/mit_b3.pth")
    model.encoder.patch_embed1.proj = nn.Conv2d(1,64,kernel_size=(7,7),stride=(4,4),padding=(3,3))
    return model