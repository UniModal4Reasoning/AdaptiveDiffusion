import argparse
import os

import numpy as np
import torch
import cv2
from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Resize
from tqdm import tqdm


def read_image(path: str):
    """
    input: path
    output: tensor (C, H, W)
    """
    img = np.asarray(Image.open(path))
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

class MultiImageDataset(Dataset):
    def __init__(self, root0, root1, is_gt=False):
        super().__init__()
        self.root0 = root0
        self.root1 = root1
        file_names0 = os.listdir(root0)
        file_names1 = os.listdir(root1)

        self.image_names0 = sorted([name for name in file_names0 if name.endswith(".jpg") or name.endswith(".png")])
        self.image_names1 = sorted([name for name in file_names1 if name.endswith(".jpg") or name.endswith(".png")])
        self.is_gt = is_gt
        assert len(self.image_names0) == len(self.image_names1)

    def __len__(self):
        return len(self.image_names0)

    def __getitem__(self, idx):
        img0 = read_image(os.path.join(self.root0, self.image_names0[idx]))
        img1 = read_image(os.path.join(self.root1, self.image_names0[idx]))
        if self.is_gt:
            # resize to 1024 x 1024
            img0 = Resize((1024, 1024))(img0)

        batch_list = [img0, img1]
        return batch_list


if __name__ == "__main__":
    psnr = PeakSignalNoiseRatio().to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--is_gt", action="store_true")
    parser.add_argument("--input_root0", type=str, required=True)
    parser.add_argument("--input_root1", type=str, required=True)
    args = parser.parse_args()

    psnr_metric = MeanMetric()
    lpips_metric = MeanMetric()
    psnr_list, lpips_list = [], []

    if args.input_root0.endswith('pt'):
        root0 = args.input_root0
        print("Loading file:", root0)
        ckpt_ori = torch.load(root0)
        all_images_ori = ckpt_ori['images']
        prompt_list = ckpt_ori['prompts']
        path0 = root0.replace('ckpt', 'img').replace('.pt', '/')
        os.makedirs(path0, exist_ok=True)
        for i, img in enumerate(all_images_ori):
            try:
                img_path = path0 + str(i) + '_' + prompt_list[i].split(",")[0].split(".")[0].split("/")[0] + '.png'
                if os.path.exists(img_path): continue
                image = to_pil_image(img, mode=None)
                image.save(img_path)
            except:
                print('Save Error' + prompt_list[i])
                pass
        args.input_root0 = path0
    
    if args.input_root1.endswith('pt'):
        root1 = args.input_root1
        print("Loading file:", root1)
        ckpt_ori = torch.load(root1)
        all_images_ori = ckpt_ori['images']
        prompt_list = ckpt_ori['prompts']
        path1 = root1.replace('ckpt', 'img').replace('.pt', '/')
        os.makedirs(path1, exist_ok=True)
        for i, img in enumerate(all_images_ori):
            try:
                img_path = path1 + str(i) + '_' + prompt_list[i].split(",")[0].split(".")[0].split("/")[0] + '.png'
                if os.path.exists(img_path): continue
                image = to_pil_image(img, mode=None)
                image.save(img_path)
            except:
                print('Save Error' + prompt_list[i])
                pass
        args.input_root1 = path1

    dataset = MultiImageDataset(args.input_root0, args.input_root1, is_gt=args.is_gt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # make a json file

    progress_bar = tqdm(dataloader)
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            batch = [img.to("cuda") for img in batch]
            batch_size = batch[0].shape[0]
            psnr_score = psnr(batch[0].to(torch.float32), batch[1].to(torch.float32))
            if not torch.isnan(psnr_score) and not torch.isinf(psnr_score):
                psnr_metric.update(psnr(batch[0].to(torch.float32), batch[1].to(torch.float32)).item(), batch_size)
                psnr_list.append(psnr_score)
            else:
                print(i)
                print("PSNR is nan or inf")
            lpips_score = lpips(batch[0] / 255, batch[1] / 255).item()
            lpips_metric.update(lpips_score, batch_size)
            lpips_list.append(lpips_score)
    fid_score = fid.compute_fid(args.input_root0, args.input_root1)

    print("PSNR:", psnr_metric.compute().item())
    print("LPIPS:", lpips_metric.compute().item())
    print("FID:", fid_score)
