import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image
import os
import time
import argparse
import numpy as np
import random
from pathlib import Path

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    if args.original:
        pipe = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16").to(f"cuda:{args.gpu}")
    else:
        import sys
        sys.path.append(Path(__file__).resolve().parent)
        from acceleration.sparse_pipeline import I2VGenXLPipeline as AdaptiveI2VGenXLPipeline
        pipe = AdaptiveI2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", 
                            torch_dtype=torch.float16, 
                            max_skip_steps=args.max_skip_steps, 
                            threshold=args.threshold, 
                            variant="fp16").to(f"cuda:{args.gpu}")
        # adaptive_pipe.scheduler = pipe.scheduler
        # pipe = adaptive_pipe

    pipe.enable_model_cpu_offload()
    
    prompt_path = f'{args.dataset}/samples.txt'
    prompt_list, img_list = [], []

    with open(prompt_path, 'r') as file:
        for img_ in file:
            img_name = img_.strip()
            img_path = f'{args.dataset}/{img_name}'
            img_list.append(img_path)
            index = img_name.split('_')[0]
            prompt_list.append(img_name.split(f'{index}_')[1].split('.png')[0])

    if args.original:
        output_dir = f'{args.output_dir}/videos-original-{args.steps}'
    else:
        output_dir = f'{args.output_dir}/videos-adaptive-{args.steps}-max_skip_step-{args.max_skip_steps}-threshold-{args.threshold}'
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (image_path, prompt) in enumerate(zip(img_list, prompt_list)):
        image = load_image(image_path)
        generator = torch.manual_seed(args.seed)

        frames = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=args.steps,
            guidance_scale=9.0,
            generator=generator
        ).frames[0]

        if args.adaptive_diffusion:
            pipe.reset_cache()
            
        video_path = export_to_gif(frames, f"{output_dir}/{prompt}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='path to the AIGCBench dataset')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--original", action="store_true")
    parser.add_argument("--adaptive_diffusion", action="store_true")

    # Adaptive Skip Setup
    parser.add_argument("--max_skip_steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.008)

    # Sampling setup
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)