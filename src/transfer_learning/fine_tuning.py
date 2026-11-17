import sys, gc, math
import logging, datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import *
from src.logger import LoggerWriter

from src.dataset_processing.dcls import MyDataset

from transformers import get_cosine_schedule_with_warmup

# 输入参数解析
parser = argparse.ArgumentParser(description="Reconstruction Test")
parser.add_argument("--gpunum", type=int, default=1, help="Number of GPUs")
parser.add_argument("--net", type=str, default='DMINet_3d_4TL', help="Initial learning rate")
parser.add_argument("--thre", type=int, default=100, help="")
parser.add_argument("--dpath", type=str, default='./entire1281288', help="Path to the training dataset")
parser.add_argument("--cp", type=str2bool, default=False, help="Whether to use checkpoint to continue uncompleted training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--wnum", type=int, default=4, help="Number of workers")
parser.add_argument("--milestone", type=int, default=15, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.00001, help="Initial learning rate")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--mfolder", type=str, default='', help="Folder to model; available when [--checkpoint] is True")
parser.add_argument("--dtype", type=str, default='2d',
                    help="2d: 2d data; 3d: 3d data")
parser.add_argument("--pname", type=str, default='', help="Name of the .pth file")

opt = parser.parse_args()

# 记录当前时间
current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
# current_time在程序一开始便记录了
log_filename = f'./logs/{opt.net}_TL_{current_time}.log'
# 设置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename)
                        , logging.StreamHandler()
                              ])
# 创建一个日志记录器
logger = logging.getLogger()
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

# 模型保存目录
save_dir = f'./trained_models/tl/{opt.net}{current_time}'

# 导入网络的模块
from importlib import import_module
if opt.dtype == '3d':
    module_name = "src.networks.3d." + opt.net
elif opt.dtype == 'tl':
    module_name = "src.networks.tl." + opt.net
net_module = import_module(module_name)
net_class = getattr(net_module, opt.net)
net = net_class()

# 采用多GPU进行预测
gpu_num = opt.gpunum
device_ids = multi_gpu(gpu_num, opt.thre)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 将 device_ids 转换为逗号分隔的字符串
cuda_visible_devices = ",".join(str(id) for id in device_ids)
# cuda_visible_devices = '7'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# device = torch.device(f"cuda:7")
device = torch.device(f"cuda:{device_ids[0]}")
model = nn.DataParallel(net, device_ids=device_ids).to(device)

# 加载参数
match opt.net:
    case 'DMINet128':
        model_path = './trained_models/pretrained/DMINet128.pth'

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        # needs to be manually freezing
        # model = load_params_partial_freezing(model_path, model, ["module.layers.0"])
    case 'TIR_3d_4TL_drop_0_1' | 'TIR_3d_3_drop_0_1':
        model_path = f'./trained_models/pretrained/{opt.pname}.pth'
        # model = load_params_partial_freezing(model_path, model, ['layers', 'blocks', 'layers'])

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    case 'TIR_3d_3':
        model_path = f'./trained_models/pretrained/{opt.pname}.pth'
        # model = load_params_partial_freezing(model_path, model, ['layers', 'blocks', 'layers'])

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)

        for param in model.parameters():
            param.requires_grad = False

        target_modules = [model.module.layers[0].residual_group.blocks[2],
                          model.module.layers[0].residual_group.blocks[3],
                          model.module.conv_after_body]
        for target_module in target_modules:
            # 将其中所有参数解冻
            for param in target_module.parameters():
                param.requires_grad = True
        print("Last two layers activated? ",
              model.get_parameter('module.layers.0.residual_group.blocks.0.layers.0.0.norm.weight').requires_grad,
              model.get_parameter('module.layers.0.residual_group.blocks.1.layers.0.0.norm.weight').requires_grad,
              model.get_parameter('module.layers.0.residual_group.blocks.2.layers.0.0.norm.weight').requires_grad,
              model.get_parameter('module.layers.0.residual_group.blocks.3.layers.0.0.norm.weight').requires_grad)
        print("TB activated? ",
              model.get_parameter('module.conv_after_body.0.weight').requires_grad)

# 损失函数
criterion = nn.MSELoss()

# 加载数据集

# 加载数据集
print('Loading dataset ...\n')
dpath = f'./dataset/' + opt.dpath + '.h5'
dataset_train = MyDataset(dataset_fname=dpath, istrain=True)
dataset_val = MyDataset(dataset_fname=dpath, istrain=False)

train_loader = DataLoader(dataset=dataset_train, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=dataset_val, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True,
                        drop_last=True)

print("## Training batches: %d" % int(len(train_loader)))
print("## Validating batches: %d" % (int(len(val_loader)) if val_loader is not None else 0))

# 对所有参数进行优化
optimizer = optim.AdamW(
    model.parameters(),
    lr=opt.lr,
    weight_decay=1e-5
)
# 设置学习率更新方式
n_training_steps = opt.epochs * (len(train_loader) // opt.batchSize)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=math.ceil(n_training_steps * 0.05),
    num_training_steps=n_training_steps   # = (training set size / batch size) * epochs
)

if opt.cp:
    read_save_dir = f'./trained_models/tl/{opt.mfolder}'
    ck_filepath = f'{read_save_dir}/checkpoint.pth'
    start_epoch, results_train, results_val, timelist = (
        load_checkpoint(model, optimizer, scheduler, ck_filepath))
    assert results_train is not None
    assert results_val is not None
    assert timelist is not None
else:
    start_epoch = 0
    results_train = {'Loss': [], 'PSNR': [], 'SSIM': []}
    results_val = {'Loss': [], 'PSNR': [], 'SSIM': []}
    timelist = []

# 训练
best = 0.
try:
    for epoch in range(start_epoch, opt.epochs):
        # print(f"Allocated memory pos1: {torch.cuda.memory_allocated(device=device) / (1024 * 1024):.4f} MB")
        print(f'Epoch-{epoch+1} begin.')
        print('Learning rate %f' % optimizer.param_groups[0]['lr'])

        model.train()  # 训练模式
        loss_list = []
        psnr_list = []
        ssim_list = []
        start_time = time.time()
        train_bar = tqdm(train_loader)
        for i, (feature, label) in enumerate(train_bar):
            label, feature = label.to(device), feature.to(device)  # 将标签、特征都放入设备中
            optimizer.zero_grad()
            predicted = model(feature)  # 获得模型输出

            loss = criterion(predicted, label)
            loss.backward()  # 反向传播，计算梯度
            psnr = batch_PSNR(predicted, label, torch.max(label))  # 计算psnr n
            # parr = predicted.cpu().detach().numpy().squeeze()
            # larr = label.cpu().detach().numpy().squeeze()
            # ssim, _ = SSIM(parr, larr, full=True, data_range=np.max(larr))  # 计算ssim
            ssim = 0.
            optimizer.step()  # 更新模型参数 n
            scheduler.step()

            if torch.isnan(loss):
                raise ValueError(f"Epoch {epoch}, Batch {i}: Loss is NaN, stopping training.")

            # 获得结果 n
            loss_list.append(loss.item())
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            # print("[Epoch %d][%d/%d] Loss: %.4f PSNR: %.4f" %
            #       (epoch + 1, i + 1, len(train_loader), loss.item(), psnr))
            train_bar.set_description(
                desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                epoch + 1, opt.epochs, loss.item(), psnr, ssim))
            # train_bar.set_description(
            #     desc='[%d/%d] Loss: %.6f PSNR: %.2f' % (
            #     epoch + 1, opt.epochs, loss.item(), psnr))

        epoch_loss = sum(loss_list) / len(loss_list)
        epoch_psnr = sum(psnr_list) / len(psnr_list)
        epoch_ssim = sum(ssim_list) / len(ssim_list)
        print("\n[Epoch %d] Training Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
              (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
        # print("\n[Epoch %d] Training Avg Loss: %.6f Avg PSNR: %.2f" %
        #       (epoch + 1, epoch_loss, epoch_psnr))
        results_train['Loss'].append(epoch_loss)
        results_train['PSNR'].append(epoch_psnr)
        results_train['SSIM'].append(epoch_ssim)

        # 验证
        model.eval()
        if val_loader != None:
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                loss_list = []
                psnr_list = []
                ssim_list = []
                for i, (feature, label) in enumerate(val_bar):
                    label, feature = label.to(device), feature.to(device)  # 将标签、特征都放入设备中
                    predicted = model(feature)  # 获得模型输出
                    loss = criterion(predicted, label)
                    psnr = batch_PSNR(predicted, label, torch.max(label))  # 计算psnr
                    parr = predicted.cpu().detach().numpy().squeeze()
                    larr = label.cpu().detach().numpy().squeeze()
                    # ssim, _ = SSIM(parr, larr, full=True, data_range=np.max(larr))  # 计算ssim
                    ssim = 0.
                    # 获得结果 n
                    loss_list.append(loss.item())
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    # print("[Epoch %d][%d/%d] Loss: %.4f PSNR: %.4f" %
                    #       (epoch + 1, i + 1, len(val_loader), loss.item(), psnr))
                    val_bar.set_description(
                        desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                            epoch + 1, opt.epochs, loss.item(), psnr, ssim))
                    # val_bar.set_description(
                    #     desc='[%d/%d] Loss: %.4f PSNR: %.4f' % (
                    #         epoch + 1, opt.epochs, loss.item(), psnr))
                epoch_loss = sum(loss_list) / len(loss_list)
                epoch_psnr = sum(psnr_list) / len(psnr_list)
                epoch_ssim = sum(ssim_list) / len(ssim_list)
                print("\n[Epoch %d] Validating Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
                      (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
                # print("\n[Epoch %d] Validating Avg Loss: %.4f Avg PSNR: %.2f" %
                #       (epoch + 1, epoch_loss, epoch_psnr))
                # 存储验证集SNR和loss
                results_val['Loss'].append(epoch_loss)
                results_val['PSNR'].append(epoch_psnr)
                results_val['SSIM'].append(epoch_ssim)
            # metric = epoch_psnr * epoch_ssim
            metric = epoch_psnr
            # metric = epoch_psnr
            if metric > best:  # 记录效果最好的模型为recon_best
                best = metric
                if not opt.cp:
                    os.makedirs(save_dir, exist_ok=True)

                torch.save(model.state_dict(),
                    os.path.join(save_dir if not opt.cp else read_save_dir,
                    'recon_best.pth'))
        # 保存模型
        if not opt.cp:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(),
            os.path.join(save_dir if not opt.cp else read_save_dir,
            f'recon{epoch + 1}.pth'))
        # 保存检查点
        save_checkpoint(model, optimizer, scheduler, epoch, timelist, results_train, results_val,
            os.path.join(save_dir if not opt.cp else read_save_dir, f'checkpoint.pth'))

        # scheduler.step()
        torch.cuda.empty_cache()
        end_time = time.time()
        timelist.append(end_time - start_time)
        print(f"[Epoch {epoch + 1}] Total time used: {end_time - start_time}s")


except ValueError as e:
    print(e)
    # 释放GPU资源
    del feature, label, predicted, loss
    torch.cuda.empty_cache()
    gc.collect()

with open(os.path.join(save_dir if not opt.cp else read_save_dir,
                           'training_info.log'), 'w') as f:
    f.write("training times:\n")
    for line in timelist:
        f.write(str(line) + "\n")
    f.write("avg_train_loss:\n")
    for line in results_train["Loss"]:
        f.write(str(line) + "\n")
    f.write("avg_train_psnr:\n")
    for line in results_train["PSNR"]:
        f.write(str(line) + "\n")
    f.write("avg_train_ssim:\n")
    for line in results_train["SSIM"]:
        f.write(str(line) + "\n")
    f.write("avg_val_loss:\n")
    for line in results_val["Loss"]:
        f.write(str(line) + "\n")
    f.write("avg_val_psnr:\n")
    for line in results_val["PSNR"]:
        f.write(str(line) + "\n")
    f.write("avg_val_ssim:\n")
    for line in results_val["SSIM"]:
        f.write(str(line) + "\n")


# 绘制第一个图：训练损失
# print(len(avg_loss_list))
plt.figure(figsize=(8, 6))
plt.plot(results_train["Loss"])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                         'training_loss.png'))
plt.close()  # 关闭当前图形，准备绘制下一个

# 绘制第二个图：训练SNR
plt.figure(figsize=(8, 6))
plt.plot(results_train["PSNR"])
plt.title('Training PSNR')
plt.xlabel('Epochs')
plt.ylabel('PSNR')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                         'training_psnr.png'))
plt.close()  # 关闭当前图形

