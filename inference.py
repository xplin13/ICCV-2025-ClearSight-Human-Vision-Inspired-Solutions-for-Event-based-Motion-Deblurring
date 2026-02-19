import os
import cv2
import numpy as np
import torch
import dataloader
from models.Model_gpu import BDHNet
from spikingjelly.activation_based import functional
from basicsr.metrics import calculate_psnr, calculate_ssim


def main():
    model = BDHNet(num_res=20)
    model = model.cuda()
    h5_filename = './dataset/GOPRO/test/'
    output_path = './output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_test = dataloader.DataLoaderVal_npz(rgb_dir=h5_filename, args=None)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=True, pin_memory=True)

    model_path = "./checkpoint/gopro/gopro_ckpt.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    count = 0
    psnr_val_rgb = []
    ssim_val_rgb = []

    with torch.no_grad():
        for batch_idx, (blur, sharp, event) in enumerate(testDataLoader):
            blur, sharp, event_frame = blur.cuda(), sharp.cuda(), event.cuda()
            output, _, mask = model(blur, event_frame)

            functional.reset_net(model)

            output = torch.clamp(output[2], 0, 1)
            mask = torch.clamp(mask, 0, 1)

            current_psnr = calculate_psnr(output*255., sharp*255., crop_border=0, input_order='HWC')
            current_ssim = calculate_ssim(output*255., sharp*255., crop_border=0, input_order='HWC')

            # 保存输出图像
            output = output.permute(0, 2, 3, 1)
            output = output.squeeze(0).cpu().detach().numpy()
            output = (output * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, f"output_{batch_idx}.png"), output)

            count += 1
            print('cnt=', count)

            ssim_val_rgb.append(current_ssim)
            psnr_val_rgb.append(current_psnr)

        avg_ssim = np.mean(ssim_val_rgb)
        avg_psnr = np.mean(psnr_val_rgb)

        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.4f}")


if __name__ == "__main__":
    main()