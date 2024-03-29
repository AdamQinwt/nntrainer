from .fc import FCBlock_v2,FCLayer
from .loss import MSE,MSEFrom0,MSEFrom1,MSEFromX,CosineLoss,FixedIDXFunc,FixedKEYFunc,WeightedSumLoss
from .trivial import UnitLayer,EmptyLayer,ActivationLayer
from .view import View,Squeeze,Mean,WindowPartition,WindowPartitionReverse,Flatten,window_partition,window_reverse
from .weight_init import model_param_stat,trunc_normal,weights_init_normal# ,model_norm_stat
from .convbase import ConvLayer,ResConvLayer,ResConvLayers,ConvBaseBlock,ResConvBaseBlock
from .model_parser import Factory,CascadedModels,parse_model,DefaultNNFactory
from .utils import DropPath,drop_path
from .fourier import FourierConv,FFC
from .transformer import SwinIR,ViT,Transformer
from .anode.ode_block import ODEBlock
from .resnet import ResNetBlock_small_pre,ResNetBlock_big,ResNetBlock_large,ResNetBlock_large_bottleneck,ResNetBlock_large_bottleneck_downsample,ResNetBlock_small,ResNetFactory
from .attention import SELayer,ResAttBlock,ResNetAttentionBlock,ResNetAttentionBlock_big,ResNetAttentionBlock_small,ResNetBlock_small_pre,AttConv2dResNetFactory,CBAMBlock,BAMBlock
from .quad_fc import QuadFCLayer
from .quad_conv import ChannelQuadLayer,calcNewLength
from .patch_discriminator.patchdis import PatchSplitMerge,ClassicDisBlock

default_factories={
    'default':DefaultNNFactory(),
    'attention':AttConv2dResNetFactory(),
    'resnet':ResNetFactory(),
}