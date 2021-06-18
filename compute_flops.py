from thop import profile
#from tc_conv_densenet import DenseNet
#from SEdensenet import DenseNet
from models import DenseNet
import torch
from models import *
from models.ghost_net import ghost_net
from models.mobilenet_v3 import MobileNetV3

if __name__ == '__main__':

    #model = ghost_net(num_classes=10)
    #model = ShuffleNetV2(net_size=1)
    #model = MobileNetV3(num_classes=10, model_type='large')

    #model = shufflenet_g2()

    # from models.mobilenet_v1 import MobileNetV1
    # model = MobileNetV1(n_class=2)

    # from shufflenet_v1 import ShuffleNet
    # model = ShuffleNet(n_groups=3, n_classes=2)
    # import torchvision
    # model = torchvision.models.shufflenet_v2_x1_0(num_classes=2)

    # from models.osdnet_hswish import OsdNet
    from models.osd_revise3 import OsdNet
    model = OsdNet(num_classes=10)

    # from models.mobilenetv2 import MobileNetV2
    # model = MobileNetV2()
    #model = MobileNet()

    # from mixnet import MixNet
    # model = MixNet(net_type='mixnet_l', num_classes=2)

    # from mobilenet_v2 import MobileNetV2
    # model = MobileNetV2(n_class=2)

    # model = DenseNet(
    #     growth_rate=12,
    #     num_classes=2,
    #     small_inputs=False,
    #     efficient=True,
    # )
    print(model)
    input = torch.randn(1, 3, 32, 32)
    #input = torch.randn(1, 3, 224, 224)
    flops,params=profile(model, inputs=(input,))
    print(flops)
    print(params)
