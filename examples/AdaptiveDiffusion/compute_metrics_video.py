import argparse
import os

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchvision.transforms import Resize
from calculate_fvd import calculate_fvd
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

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def read_video(path: str):
    """
    input: path
    output: tensor (L, C, H, W)
    """
    video = cv2.VideoCapture(path)
    frames = [x for x in _frame_from_video(video)]
    frames_rgb = []
    for i in range(len(frames)):
        converted = cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
        frames_rgb.append(converted[None, :, :, :])
    frames = torch.from_numpy(np.concatenate(frames_rgb, axis=0)).permute(0, 3, 1, 2)
    return frames

class MultiVideoDataset(Dataset):
    def __init__(self, root0, root1, is_gt=False):
        super().__init__()
        self.root0 = root0
        self.root1 = root1
        file_names0 = os.listdir(root0)
        file_names1 = os.listdir(root1)

        self.video_names0 = sorted([name for name in file_names0 if name.endswith(".mp4") or name.endswith(".gif")])
        self.video_names1 = sorted([name for name in file_names1 if name.endswith(".mp4") or name.endswith(".gif")])
        self.is_gt = is_gt
        assert len(self.video_names0) == len(self.video_names1)

    def __len__(self):
        return len(self.video_names0)

    def __getitem__(self, idx):
        vid0 = read_video(os.path.join(self.root0, self.video_names0[idx]))
        vid1 = read_video(os.path.join(self.root1, self.video_names1[idx]))

        batch_list = [vid0, vid1]
        return batch_list


if __name__ == "__main__":
    psnr = PeakSignalNoiseRatio().to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--is_gt", action="store_true")
    parser.add_argument("--input_root0", type=str, required=True)
    parser.add_argument("--input_root1", type=str, required=True)
    args = parser.parse_args()

    psnr_metric = MeanMetric()
    lpips_metric = MeanMetric()
    fvd_metric = MeanMetric()
    fvd_list = []

    dataset = MultiVideoDataset(args.input_root0, args.input_root1, is_gt=args.is_gt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # make a json file

    progress_bar = tqdm(dataloader)
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            # to cuda
            batch = [img.to("cuda") for img in batch]
            batch_size = batch[0].shape[0]
            batch0, batch1 = batch[0].view(*batch[0].shape[1:]), batch[1].view(*batch[1].shape[1:])
            psnr_score = psnr(batch0.to(torch.float32), batch1.to(torch.float32))
            if not torch.isnan(psnr_score) and not torch.isinf(psnr_score):
                psnr_metric.update(psnr(batch0.to(torch.float32), batch1.to(torch.float32)).item(), batch_size)
            else:
                print(i)
                print("PSNR is nan or inf")
            lpips_metric.update(lpips(batch0 / 255, batch1 / 255).item(), batch_size)
            fvd_score = calculate_fvd((batch[0] / 255).to(torch.float32), (batch[1] / 255).to(torch.float32), torch.device("cuda"))
            fvd_score = fvd_score['value'][batch[0].shape[1]]
            fvd_metric.update(fvd_score, batch_size)
            fvd_list.append(fvd_score)
    print("PSNR:", psnr_metric.compute().item())
    print("LPIPS:", lpips_metric.compute().item())
    print("FVD:", fvd_metric.compute().item())