from thop import profile
import torch
from models import *
from models.shufflenet_v1 import shufflenet_g2
from models.shufflenet_v2 import ShuffleNetV2
from models.mobilenet_v1 import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.mobilenet_v3 import MobileNetV3
from models.ghost_net import ghost_net
from models.gather_net import GatherNet

if __name__ == '__main__':

    # model = ghost_net(num_classes=10)
    # model = shufflenet_g2()
    # model = ShuffleNetV2(net_size=1)
    # model = MobileNet(n_class=10)
    # model = MobileNetV2(n_class=10)
    # model = MobileNetV3(num_classes=10, model_type='small')
    # model = MobileNetV3(num_classes=10, model_type='large')
    model = GatherNet(num_classes=10)

    print(model)
    input = torch.randn(1, 3, 32, 32)
    flops,params=profile(model, inputs=(input,))
    print(flops)
    print(params)
