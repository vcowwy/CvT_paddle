import collections

import torch
import paddle


def torch2paddle():
    torch_path = "CvT_torch.pth"
    paddle_path = "CvT_tpd.pdparams"
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    paddle_state_dict = {}

    fc_names = ["mlp.fc1.weight", "mlp.fc2.weight", "head.weight",
                "attn.proj_q.weight", "attn.proj_k.weight",
                "attn.proj_v.weight", "attn.proj.weight"]

    for k, v in torch_state_dict.items():
        flag = [i in k for i in fc_names]
        v = v.detach().cpu().numpy()
        if any(flag):
            v = v.transpose()
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        paddle_state_dict[k] = v

    print(paddle_state_dict.keys())
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
