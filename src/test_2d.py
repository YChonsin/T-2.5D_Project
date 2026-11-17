'''
2d数据重建任务的=测试程序=
'''
import glob
import torch
import re
import ast
import numpy as np

import os, sys
sys.path.insert(0, os.getcwd())
from src.utils import *
import scipy.io as sio

parser = argparse.ArgumentParser(description="Reconstruction Test")
parser.add_argument("--mtype", type=str, default='regular',
                    help='missing type; regular: delete one out of every three; irregular: randomly delete 50% of traces, patch: patch-wise missing')
parser.add_argument("--mrate", type=float, default='0.0', help='missing rate')
parser.add_argument("--urf", type=int, default='0', help='uniform random factor, reused as patch size for --mtype patch')
parser.add_argument("--folder", type=str, default='./pre_data/test', help='test data folder')
parser.add_argument("--moname", type=str, default=0, help='filename of the saved model, without .pth')
parser.add_argument("--net", type=str, default='MambaIR', help="Initial learning rate")
parser.add_argument("--gpunum", type=int, default=1, help="Number of GPUs")
parser.add_argument("--shape", type=int, nargs=2, default=128, help="Size of patch")
parser.add_argument("--testtype", type=int, default=1, help="")
parser.add_argument("--mfolder", type=str, default='0000', help="**Must: the folder of the trained model")
parser.add_argument("--thre", type=int, default=100, help="")
parser.add_argument("--tmp", type=bool, default=False, help="")
parser.add_argument("--tl", type=bool, default=False, help="")
parser.add_argument("--tlfolder", type=str, default="", help="")
parser.add_argument("--norm", type=lambda value: 'TPNN' if value == 'asinhstd' else value,
                    choices=['absmax', 'log1p', 'log1pmax', 'TPNN', 'asinh', 'none'], default="absmax",
                    help="Normalization; asinhstd is accepted as a legacy alias of TPNN")
parser.add_argument("--input_mode", choices=['masking', 'real'], default='masking',
                    help="masking: delete traces from complete data; real: reconstruct existing all-zero traces without reference metrics")
parser.add_argument("--p_layers", type=str, default='[]',
                    help="")
parser.add_argument("--ordered", type=bool, default=False, help="")
parser.add_argument("--dkey", type=str, default='', help="")

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 导入网络的模块
    from importlib import import_module

    module_name = "src.networks.2d." + opt.net

    net_module = import_module(module_name)
    net_class = getattr(net_module, opt.net)
    net_type = opt.net
    # 加载模型
    net = net_class()
    print("Loading model: ", opt.net)

    # 采用多GPU进行预测
    gpu_num = opt.gpunum
    device_ids = multi_gpu(gpu_num, opt.thre)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 将 device_ids 转换为逗号分隔的字符串
    cuda_visible_devices = ",".join(str(id) for id in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(net, device_ids=device_ids)
    # model = net

    if opt.tl:
        save_dir = f'./trained_models/{opt.tlfolder}/2d/{opt.mfolder}'
    else:
        save_dir = f'./trained_models/2d/{opt.mfolder}'

    model_filename = 'recon_best.pth' if opt.moname == '' else f'{opt.moname}.pth'
    model.load_state_dict(torch.load(os.path.join(save_dir, model_filename), map_location="cpu"))
    model = model.to(device)

    model.eval()  # 评估模式
    print('Loading data info ...\n')
    # print(os.path.join('./' + opt.folder))
    files_source = glob.glob(os.path.join(f'./data/{opt.folder}', '*'))
    files_source = sorted(files_source) # 排序
    # print(files_source)
    assert len(files_source) != 0
    # files_source = glob.glob(os.path.join('./sdata/test_recon', 'field_label.mat'))
    file_to_metrics_map = {}
    reconstructed_map = {}
    arr_map = {}
    miss_arr_map = {}
    combine_order = {}
    inference_times = {}
    with ((torch.no_grad())):

        start_time_all = time.time()

        seed_dict = []
        for i in range(len(files_source)):
            seed_dict.append((i + 1) * 666)

        print(seed_dict)
        # 处理数据
        for file_i, f in enumerate(files_source):

            block_shape = tuple(opt.shape)

            fname = os.path.basename(f)

            if opt.ordered:
                nums_in_fname = re.findall(r'\d+', fname)
                last_num = int(nums_in_fname[-1]) - 1
                combine_order[fname] = last_num

            start_time = time.time()

            print(f"\nFile \'{fname}\' starting deleting and reconstructing\n")
            from os.path import splitext, basename
            arr = load_data_auto(f, opt.dkey) # 读取mat数据，以及其中的数组

            arr_max = np.max(np.abs(arr)) or 1.
            arr_mean = np.mean(arr)
            arr_std = np.std(arr)
            n_rows, n_cols = arr.shape
            print(arr.shape)
            print(f"Missing rate: {opt.mrate}")



            # 模拟不同类型的缺失
            if opt.mtype == 'regular':  # 常规缺失：每2道删除1道；
                miss_arr = arr.copy()
                match opt.mrate:
                    case 0.33:
                        for col in range(1, miss_arr.shape[1], 3):
                            miss_arr[:, col] = 0
                    case 0.5:
                        for col in range(1, miss_arr.shape[1], 2):
                            miss_arr[:, col] = 0
                    case 0.67:
                        for col in range(1, miss_arr.shape[1], 3):
                            miss_arr[:, col:col + 2] = 0
                    case 0.75:
                        for col in range(1, miss_arr.shape[1], 4):
                            miss_arr[:, col:col + 3] = 0
                    case 0.8:
                        for col in range(1, miss_arr.shape[1], 5):
                            miss_arr[:, col:col + 4] = 0
                    case 0.83:
                        for col in range(1, miss_arr.shape[1], 6):
                            miss_arr[:, col:col + 5] = 0
                    case 0.86:
                        for col in range(1, miss_arr.shape[1], 7):
                            miss_arr[:, col:col + 6] = 0
                    case 0.88:
                        for col in range(1, miss_arr.shape[1], 8):
                            miss_arr[:, col:col + 7] = 0
                    case 0.89:
                        for col in range(1, miss_arr.shape[1], 9):
                            miss_arr[:, col:col + 8] = 0
                    case 0.9:
                        for col in range(1, miss_arr.shape[1], 10):
                            miss_arr[:, col:col + 9] = 0
                    case _:
                        mrate = [0.33, 0.5, 0.67, 0.75, 0.8, 0.83, 0.86, 0.88, 0.89, 0.9]
                        raise ValueError(f"{opt.mrate} is an unsupported missing rate. \n"
                                         f"Supported rates are: {mrate}")

                mask = miss_arr != 0

            elif opt.mtype == 'irregular':  # 非常规缺失：随机删除道数。为了复现，人工设置随机种子
                if opt.urf != 0:
                    miss_arr, mask = remove_traces(arr, opt.mrate, opt.mrate,
                                 'ur', ur_factor=opt.urf, seed=seed_dict[file_i])
                else:
                    miss_arr, mask = remove_traces(arr, opt.mrate, opt.mrate,
                                             'r')

            elif opt.mtype == 'patch':
                miss_arr, mask = remove_patches(arr, opt.urf, opt.mrate, opt.mrate)
            elif opt.mtype == 'none':
                miss_arr = arr.copy()
                mask = miss_arr != 0

            # # 归一化
            match opt.norm:
                case 'absmax':
                    arr = arr / arr_max
                    miss_arr = miss_arr / arr_max
                case 'log1p':
                    arr = norm_by_log(arr)
                    miss_arr = norm_by_log(miss_arr)
                case 'log1pmax':
                    arr, arr_log_max = norm_by_log_max(arr)
                    miss_arr = norm_by_log(miss_arr) / arr_log_max
                case 'TPNN':
                    arr, _, _ = norm_by_arcsinh_std(arr, arr_mean, arr_std)
                    miss_arr, _, _ = norm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                    print("arr and miss_arr normed by TPNN")
                case 'asinh':
                    arr = norm_by_arcsinh(arr)
                    miss_arr = norm_by_arcsinh(miss_arr)
                case _:
                    pass

            arr = pad(arr, *block_shape)
            miss_arr = pad(miss_arr, *block_shape)
            print("After padding: ", arr.shape)
            inference_time = 0.

            def predict_func(data, p_layers):
                nonlocal inference_time
                if data.is_cuda:
                    torch.cuda.synchronize(data.device)
                forward_start = time.perf_counter()

                if len(p_layers) > 0:
                    output = model(data, p_layers)  # 获得模型输出
                else:
                    output = model(data)
                # output = model(data, p_layers)

                if data.is_cuda:
                    torch.cuda.synchronize(data.device)
                inference_time += time.perf_counter() - forward_start
                return output

            if opt.testtype == 0:  # 直接输入整幅，适用于CNN-based

                with torch.no_grad():  # this can save much memory
                    miss_arr = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    p_layers = ast.literal_eval(opt.p_layers)
                    reconstructed = predict_func(miss_arr, p_layers).cpu().detach().numpy().squeeze()
                miss_arr = miss_arr.cpu().detach().numpy().squeeze()
                # # debug
                # sio.savemat("./results/intermediate1.mat", {'reconstructed': reconstructed})
                # # debug end

                match opt.norm:
                    case 'absmax':
                        arr = arr * arr_max
                        miss_arr = miss_arr * arr_max
                        reconstructed = reconstructed * arr_max
                    case 'log1p':
                        arr = denorm_by_log(arr)
                        miss_arr = denorm_by_log(miss_arr)
                        reconstructed = denorm_by_log(reconstructed)
                    case 'log1pmax':
                        arr = denorm_by_log_max(arr, arr_log_max)
                        miss_arr = denorm_by_log_max(miss_arr, arr_log_max)
                        reconstructed = denorm_by_log_max(reconstructed, arr_log_max)
                    case 'TPNN':
                        arr = denorm_by_arcsinh_std(arr, arr_mean, arr_std)
                        miss_arr = denorm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                        reconstructed = denorm_by_arcsinh_std(reconstructed, arr_mean, arr_std)
                    case 'asinh':
                        arr = denorm_by_arcsinh(arr)
                        miss_arr = denorm_by_arcsinh(miss_arr)
                        reconstructed = denorm_by_arcsinh(reconstructed)
                    case _:
                        pass

                arr = arr[:n_rows, :n_cols]
                miss_arr = miss_arr[:n_rows, :n_cols]
                reconstructed = reconstructed[:n_rows, :n_cols]

            elif opt.testtype == 1:  # (平滑处理)分片预测，然后结合
                with torch.no_grad():  # this can save much memory
                    print("Predicting...")
                    p_layers = ast.literal_eval(opt.p_layers)

                    reconstructed = test_with_sliding_window_2d_for_priors(
                        miss_arr, block_shape, predict_func, device, tuple(x // 2 for x in block_shape),
                        p_layers)
                    # # debug
                    # sio.savemat("./results/intermediate1.mat", {'reconstructed': reconstructed})
                    # # debug end
                print(f"reconstructed.shape: {reconstructed.shape}")
                match opt.norm:
                    case 'absmax':
                        arr = arr * arr_max
                        miss_arr = miss_arr * arr_max
                        reconstructed = reconstructed * arr_max
                    case 'log1p':
                        arr = denorm_by_log(arr)
                        miss_arr = denorm_by_log(miss_arr)
                        reconstructed = denorm_by_log(reconstructed)
                    case 'log1pmax':
                        arr = denorm_by_log_max(arr, arr_log_max)
                        miss_arr = denorm_by_log_max(miss_arr, arr_log_max)
                        reconstructed = denorm_by_log_max(reconstructed, arr_log_max)
                    case 'TPNN':
                        arr = denorm_by_arcsinh_std(arr, arr_mean, arr_std)
                        miss_arr = denorm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                        reconstructed = denorm_by_arcsinh_std(reconstructed, arr_mean, arr_std)
                        print("arr, miss_arr, and reconstructed denormed by TPNN")
                    case 'asinh':
                        arr = denorm_by_arcsinh(arr)
                        miss_arr = denorm_by_arcsinh(miss_arr)
                        reconstructed = denorm_by_arcsinh(reconstructed)
                    case _:
                        pass

                arr = arr[:n_rows, :n_cols]
                miss_arr = miss_arr[:n_rows, :n_cols]
                reconstructed = reconstructed[:n_rows, :n_cols]

            else:
                raise NotImplementedError(f"Testtype {opt.testtype} not implemented.")

            # # debug
            # sio.savemat("./results/intermediate2.mat", {'reconstructed': reconstructed})
            # # debug end
            # 进行回填操作
            # print(mask.min(), mask.max(), mask.mean())
            reconstructed, difference = refill(arr=arr,
                                               reconstructed=reconstructed,
                                               mask=mask)

            arr_map[fname] = arr
            miss_arr_map[fname] = miss_arr
            reconstructed_map[fname] = reconstructed
            inference_times[fname] = inference_time if opt.testtype in (0, 1) else None
            # 保存为mat文件

            dtype = '2d_tmp' if opt.tmp else '2d'
            result_dir = f'./results/{dtype}/{net_type}/{os.path.basename(opt.folder)}/'

            os.makedirs(result_dir, exist_ok=True)
            result = {'complete': arr, 'missing': miss_arr,
                          'reconstructed': reconstructed, 'difference': reconstructed - arr}
            sio.savemat(result_dir + f"{fname[:-4]}.mat", result)
            print(f"Results saved to {result_dir + f"{fname[:-4]}.mat"}")

            if opt.input_mode == 'real':
                file_to_metrics_map[fname] = None
                print(f"{fname}: reference metrics unavailable for real missing data")
                print(f"Model forward time of {fname}: {inference_time}s")
                print(f"Processing time of {fname}: {time.time() - start_time}s")
                continue

            # 衡量数值性能
            snr_miss = array_SNR(miss_arr, arr)
            psnr_miss = array_PSNR(miss_arr, arr)
            rmse_miss = array_RMSE(miss_arr, arr)
            ssim_miss = array_SSIM(miss_arr, arr)

            snr_test = array_SNR(reconstructed, arr)
            psnr_test = array_PSNR(reconstructed, arr)
            rmse_test = array_RMSE(reconstructed, arr)
            ssim_test = array_SSIM(reconstructed, arr)

            end_time = time.time()

            metrics = [snr_miss, psnr_miss, rmse_miss, ssim_miss,
                       snr_test, psnr_test, rmse_test, ssim_test,
                       end_time - start_time]
            file_to_metrics_map.update({fname: metrics})
            print(f"SNR on missing {os.path.basename(f)}: {snr_miss: .2f}")
            print(f"PSNR on missing {os.path.basename(f)}: {psnr_miss: .2f}")
            print(f"RMSE on missing {os.path.basename(f)}:  {rmse_miss: .4f}")
            print(f"SSIM on missing {os.path.basename(f)}: {ssim_miss: .6f}")
            print(f"SNR on reconstructed {os.path.basename(f)}: {snr_test: .2f}")
            print(f"PSNR on reconstructed {os.path.basename(f)}: {psnr_test: .2f}")
            print(f"RMSE on reconstructed {os.path.basename(f)}:  {rmse_test: .4f}")
            print(f"SSIM on reconstructed {os.path.basename(f)}: {ssim_test: .6f}")

            if inference_times[fname] is not None:
                print(f"Model forward time of {fname}: {inference_time}s")
            print(f"Processing time of {fname}: {end_time - start_time}s")

        end_time_all = time.time()

    with open(os.path.join(result_dir, 'metrics.log'), 'w') as result_file:
        print("----------------------------------------------------------")
        for key, value in file_to_metrics_map.items():
            if inference_times[key] is not None:
                result_file.write(f"Model forward time of {key}: {inference_times[key]}s\n")
            if value is None:
                result_file.write(f"{key}: reference metrics unavailable for real missing data\n")
                continue
            print(f"SNR on missing {key}: {value[0]: .2f}")
            print(f"PSNR on missing {key}: {value[1]: .2f}")
            print(f"RMSE on missing {key}:  {value[2]: .4f}")
            print(f"SSIM on missing {key}: {value[3]: .6f}")
            print(f"SNR on reconstructed {key}: {value[4]: .2f}")
            print(f"PSNR on reconstructed {key}: {value[5]: .2f}")
            print(f"RMSE on reconstructed {key}:  {value[6]: .4f}")
            print(f"SSIM on reconstructed {key}: {value[7]: .6f}")
            print("")
            result_file.write(f"SNR on missing {key}: {value[0]: .2f}\n")
            result_file.write(f"PSNR on missing {key}: {value[1]: .2f}\n")
            result_file.write(f"RMSE on missing {key}:  {value[2]: .4f}\n")
            result_file.write(f"SSIM on missing {key}: {value[3]: .6f}\n")
            result_file.write(f"SNR on reconstructed {key}: {value[4]: .2f}\n")
            result_file.write(f"PSNR on reconstructed {key}: {value[5]: .2f}\n")
            result_file.write(f"RMSE on reconstructed {key}:  {value[6]: .4f}\n")
            result_file.write(f"SSIM on reconstructed {key}: {value[7]: .6f}\n")
            # print(f"Inference time of {key}: {value[6]}s")

        if len(combine_order) != 0:
            names = sorted(combine_order, key=lambda name: (combine_order[name], name))
            new_arr = np.stack([arr_map[name] for name in names], axis=2)
            new_miss_arr = np.stack([miss_arr_map[name] for name in names], axis=2)
            new_reconstructed = np.stack([reconstructed_map[name] for name in names], axis=2)

            if opt.input_mode == 'real':
                combined_mask = np.broadcast_to(np.any(new_miss_arr != 0, axis=0, keepdims=True), new_miss_arr.shape).copy()
                sio.savemat(result_dir + "combined.mat",
                            {'missing': new_miss_arr, 'reconstructed': new_reconstructed,
                             'mask': combined_mask, 'source_files': names})
                result_file.write("combined: reference metrics unavailable for real missing data\n")
                print(f"Combined result saved to {result_dir + 'combined.mat'}")
                print(f"Processing time for all files: {end_time_all - start_time_all}s")
                return

            print(new_miss_arr.shape)
            snr_miss = array_SNR(new_miss_arr, new_arr)
            psnr_miss = array_PSNR(new_miss_arr, new_arr)
            rmse_miss = array_RMSE(new_miss_arr, new_arr)
            ssim_miss = np.mean([array_SSIM(miss_arr_map[name], arr_map[name]) for name in names])


            snr_test = array_SNR(new_reconstructed, new_arr)
            psnr_test = array_PSNR(new_reconstructed, new_arr)
            rmse_test = array_RMSE(new_reconstructed, new_arr)
            ssim_test = np.mean([array_SSIM(reconstructed_map[name], arr_map[name]) for name in names])

            sio.savemat(result_dir + "combined.mat",
                        {'complete': new_arr, 'missing': new_miss_arr,
                          'reconstructed': new_reconstructed, 'difference': new_reconstructed - new_arr,
                          'source_files': names})

            print(f"Combined result saved to {result_dir + "combined.mat"}")

            print("--------------------combined-------------------------")
            print(f"SNR on missing combined: {snr_miss: .2f}")
            print(f"PSNR on missing combined: {psnr_miss: .2f}")
            print(f"RMSE on missing combined:  {rmse_miss: .4f}")
            print(f"SSIM on missing combined: {ssim_miss: .6f}")
            print(f"SNR on reconstructed combined: {snr_test: .2f}")
            print(f"PSNR on reconstructed combined: {psnr_test: .2f}")
            print(f"RMSE on reconstructed combined:  {rmse_test: .4f}")
            print(f"SSIM on reconstructed combined: {ssim_test: .6f}")
            print("----------------------------------------------------------")
            result_file.write("--------------------combined-------------------------\n")
            result_file.write(f"SNR on missing combined: {snr_miss: .2f}\n")
            result_file.write(f"PSNR on missing combined: {psnr_miss: .2f}\n")
            result_file.write(f"RMSE on missing combined:  {rmse_miss: .4f}\n")
            result_file.write(f"SSIM on missing combined: {ssim_miss: .6f}\n")
            result_file.write(f"SNR on reconstructed combined: {snr_test: .2f}\n")
            result_file.write(f"PSNR on reconstructed combined: {psnr_test: .2f}\n")
            result_file.write(f"RMSE on reconstructed combined:  {rmse_test: .4f}\n")
            result_file.write(f"SSIM on reconstructed combined: {ssim_test: .6f}\n")
            result_file.write("----------------------------------------------------------")

        print(f"Processing time for all files: {end_time_all - start_time_all}s")

if __name__ == "__main__":
    main()
