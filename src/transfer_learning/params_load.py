import torch
import torch.nn as nn
from collections import Counter
# from networks.DMINet import DMINet128
# checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}  # 去掉 module.


def all_in_name_exact(keywords, target_str):
    lst_count = Counter(keywords)  # 统计 list 中每个元素的出现次数

    # 检查 list 里的每个元素是否在 name 里出现了足够的次数
    for item, count in lst_count.items():
        if target_str.count(item) < count:
            return False
    return True

def load_specific_params(path,
                         model,
                         keywords: list):
    checkpoint = torch.load(path, map_location="cpu")
    # 取得self_attention的部分，只将它们载入模型
    params_to_load = {k: v for k, v in checkpoint.items() if all_in_name_exact(keywords, k)}
    print(params_to_load.keys())
    # strict=False，这样只会加载匹配的参数，不影响其他层
    model.load_state_dict(params_to_load, strict=False)

    # 冻结keyword层，其他层解冻
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.module.named_parameters():

        if 'module.' + name in params_to_load.keys():
            print("Freezing:", name)
            param.requires_grad = False

    return model
def load_specific_params_4_TIR3(path,
                         model,
                         keywords: list):
    checkpoint = torch.load(path, map_location="cpu")
    # 取得self_attention的部分，只将它们载入模型
    params_to_load = {k: v for k, v in checkpoint.items() if all_in_name_exact(keywords, k)}
    print(params_to_load.keys())
    # strict=False，这样只会加载匹配的参数，不影响其他层
    model.load_state_dict(params_to_load, strict=False)

    # 冻结keyword层，其他层解冻
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.module.named_parameters():
        print(name)
        if 'module.' + name in params_to_load.keys() and "blocks.3" not in name:
            print("Freezing:", name)
            param.requires_grad = False

    return model

def load_params_partial_freezing(path,
                         model,
                         keywords: list):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint)

    # 取得需要冻结的部分
    params_to_freeze = {k: v for k, v in checkpoint.items() if all_in_name_exact(keywords, k)}

    print("Freezing:", params_to_freeze.keys())
    # 冻结keyword层，其他层解冻
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.module.named_parameters():
        if name in params_to_freeze.keys():
            print("Freezing:", name)
            param.requires_grad = False
            print("Frozen?", param.requires_grad)

    return model

def load_params_4UNet1(path,
                model):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    # 冻结所有层，然后解冻stadapter层
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.module.named_parameters():
        if 'stadapter' in name:
            param.requires_grad = True
    return model


if __name__ == "__main__":
    # model = TIR_3d_4TL_()
    device = torch.device("cuda:0")
    model = nn.DataParallel(model, device_ids=[0]).to(device)
    # model_path = './trained_models/pretrained/TIR_3.pth'
    model_path = './trained_models/pretrained/for_fine_tuning/TIR_3d_4TL.pth'
    # model = load_specific_params(model_path, model, keywords=['blocks', 'layers', 'layers'])
    model = load_params_partial_freezing(model_path, model, keywords=['blocks', 'layers', 'layers'])
    # model = load_specific_params_4_TIR3(model_path, model, keywords=['blocks', 'layers', 'layers'])


    print(model.get_parameter('module.layers.0.residual_group.last_block.layers.0.0.norm.weight').requires_grad)
    x = torch.randn(1, 1, 40, 40, 16).to(device)
    pred = model(x)

    # 损失函数
    criterion = nn.MSELoss()

    with torch.cuda.amp.autocast():
        loss = criterion(pred, x)
        loss.backward()  # 反向传播，计算梯度
    print(pred.shape)
