import numpy as np
import scipy.io as sio

import os, time

from src.utils import load_data_auto, remove_traces_3d, array_SNR, array_PSNR, array_RMSE
from skimage.metrics import structural_similarity as ssim

def fk_pocs_3d(data, mask, niter=50, threshold_ratio=0.8):
    """
    3D F-K POCS interpolation

    Parameters
    ----------
    data : ndarray
        Shape (nx, ny, nt), missing traces filled with 0.

    mask : ndarray
        Same shape as data.
        Known samples = 1, missing samples = 0.

    niter : int
        Number of POCS iterations.

    threshold_ratio : float
        Initial threshold ratio.

    Returns
    -------
    rec : ndarray
        Interpolated data.
    """
    rec = data.copy()

    for k in range(niter):
        print(f"Iteration: {k}")
        # Forward FFT
        F = np.fft.fftn(rec)

        # Linearly decreasing threshold
        lam = threshold_ratio * np.max(np.abs(F)) * (1.0 - k / niter)

        # Soft thresholding
        # mag = np.abs(F)
        # F = np.where(
        #     mag > lam,
        #     F * (mag - lam) / (mag + 1e-12),
        #     0
        # )
        F[np.abs(F) < lam] = 0

        # Inverse FFT
        rec_tmp = np.real(np.fft.ifftn(F))

        # Data consistency
        rec = mask * data + (1 - mask) * rec_tmp

    return rec


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='recon_training')
    parser.add_argument("--df", type=str, default="",
                        help="Path to the test data")
    opt = parser.parse_args()
    import glob
    fnames = glob.glob(f'./data/{opt.df}/' + '*.mat')
    result_dir = "./results/3d/fk_POCS_3d"

    for fname in fnames:
        arr = load_data_auto(fname, "")
        seed_dict = []
        for i in range(arr.shape[2]):
            seed_dict.append((i + 1) * 666)

        print(seed_dict)

        start_time = time.time()
        miss_arr, mask = remove_traces_3d(arr, 0.5, 0.5,
                  'ur', ur_factor=4, seed_dict=seed_dict)

        reconstructed = fk_pocs_3d(miss_arr, mask=mask, niter=300)
        os.makedirs(result_dir, exist_ok=True)
        sio.savemat(os.path.join(result_dir, f'{os.path.basename(fname)}'),
                    {'complete': arr, 'missing': miss_arr,
                     'reconstructed': reconstructed, 'difference': reconstructed - arr})

        arr_max = np.max(np.abs(arr))
        snr_miss = array_SNR(miss_arr, arr)
        psnr_miss = array_PSNR(miss_arr, arr)
        nrmse_miss = array_RMSE(miss_arr, arr)
        ssim_miss, _ = ssim(miss_arr, arr, full=True, data_range=arr_max)

        snr_test = array_SNR(reconstructed, arr)
        psnr_test = array_PSNR(reconstructed, arr)
        nrmse_test = array_RMSE(reconstructed, arr)
        ssim_test, _ = ssim(reconstructed, arr, full=True, data_range=arr_max)

        end_time = time.time()
        print(f"Processing time of {fname}: {end_time - start_time}s")
        with open(os.path.join(result_dir, 'metrics.log'), 'w') as result_file:
            print(f"SNR on missing {os.path.basename(fname)}: {snr_miss: .2f}")
            print(f"PSNR on missing {os.path.basename(fname)}: {psnr_miss: .2f}")
            print(f"NRMSE on missing {os.path.basename(fname)}:  {nrmse_miss: .4f}")
            print(f"SSIM on missing {os.path.basename(fname)}: {ssim_miss: .6f}")
            print(f"SNR on reconstructed {os.path.basename(fname)}: {snr_test: .2f}")
            print(f"PSNR on reconstructed {os.path.basename(fname)}: {psnr_test: .2f}")
            print(f"NRMSE on reconstructed {os.path.basename(fname)}:  {nrmse_test: .4f}")
            print(f"SSIM on reconstructed {os.path.basename(fname)}: {ssim_test: .6f}")
            result_file.write(f"SNR on missing {os.path.basename(fname)}:  {snr_miss: .2f}\n")
            result_file.write(f"PSNR on missing {os.path.basename(fname)}:  {psnr_miss: .2f}\n")
            result_file.write(f"NRMSE on missing {os.path.basename(fname)}:  {nrmse_miss: .4f}\n")
            result_file.write(f"SSIM on missing {os.path.basename(fname)}: {ssim_miss: .6f}\n")
            result_file.write(f"SNR on reconstructed {os.path.basename(fname)}:  {snr_test: .2f}\n")
            result_file.write(f"PSNR on reconstructed {os.path.basename(fname)}:  {psnr_test: .2f}\n")
            result_file.write(f"NRMSE on reconstructed {os.path.basename(fname)}:  {nrmse_test: .4f}\n")
            result_file.write(f"SSIM on reconstructed {os.path.basename(fname)}: {ssim_test: .6f}\n")

            for i in range(0, arr.shape[2]):
                snr_miss = array_SNR(miss_arr[:, :, i], arr[:, :, i])
                psnr_miss = array_PSNR(miss_arr[:, :, i], arr[:, :, i])
                nrmse_miss = array_RMSE(miss_arr[:, :, i], arr[:, :, i])
                ssim_miss, _ = ssim(miss_arr[:, :, i], arr[:, :, i], full=True, data_range=np.max(np.abs(arr[:, :, i])))

                snr_test = array_SNR(reconstructed[:, :, i], arr[:, :, i])
                psnr_test = array_PSNR(reconstructed[:, :, i], arr[:, :, i])
                nrmse_test = array_RMSE(reconstructed[:, :, i], arr[:, :, i])
                ssim_test, _ = ssim(reconstructed[:, :, i], arr[:, :, i], full=True,
                                    data_range=np.max(np.abs(arr[:, :, i])))
                print(f"SNR on missing {os.path.basename(fname)}, slice {i}:  {snr_miss: .2f}")
                print(f"PSNR on missing {os.path.basename(fname)}, slice {i}:  {psnr_miss: .2f}")
                print(f"NRMSE on missing {os.path.basename(fname)}, slice {i}:  {nrmse_miss: .4f}")
                print(f"SSIM on missing {os.path.basename(fname)}, slice {i}: {ssim_miss: .6f}")
                print(f"SNR on reconstructed {os.path.basename(fname)}, slice {i}:  {snr_test: .2f}")
                print(f"PSNR on reconstructed {os.path.basename(fname)}, slice {i}:  {psnr_test: .2f}")
                print(f"NRMSE on reconstructed {os.path.basename(fname)}, slice {i}:  {nrmse_test: .4f}")
                print(f"SSIM on reconstructed {os.path.basename(fname)}, slice {i}: {ssim_test: .6f}")
                print()
                result_file.write(f"SNR on missing {os.path.basename(fname)}, slice {i}:  {snr_miss: .2f}\n")
                result_file.write(f"PSNR on missing {os.path.basename(fname)}, slice {i}:  {psnr_miss: .2f}\n")
                result_file.write(f"NRMSE on missing {os.path.basename(fname)}, slice {i}:  {nrmse_miss: .4f}\n")
                result_file.write(f"SSIM on missing {os.path.basename(fname)}, slice {i}: {ssim_miss: .6f}\n")
                result_file.write(f"SNR on reconstructed {os.path.basename(fname)}, slice {i}:  {snr_test: .2f}\n")
                result_file.write(f"PSNR on reconstructed {os.path.basename(fname)}, slice {i}:  {psnr_test: .2f}\n")
                result_file.write(f"NRMSE on reconstructed {os.path.basename(fname)}, slice {i}:  {nrmse_test: .4f}\n")
                result_file.write(f"SSIM on reconstructed {os.path.basename(fname)}, slice {i}: {ssim_test: .6f}\n")