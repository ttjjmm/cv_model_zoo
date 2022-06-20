import paddle, os
import torch
from collections import OrderedDict


def convert2torch(model_path):
    state_dict = paddle.load(model_path)
    torch_state_dict = OrderedDict()

    for k, v in state_dict.items():
        torch_state_dict[k] = torch.tensor(v.numpy(), dtype=torch.float32)

    filename = model_path.split('/')[-1].split('.')[0]
    torch.save(torch_state_dict, os.path.join('./', '{}.pth'.format(filename)))


if __name__ == '__main__':
    convert2torch('/home/tjm/Downloads/PPLCNetV2_base_pretrained.pdparams')



