
import time
import argparse
import numpy as np
import random

import os
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    if args.dataset == 'parti':
        prompts = load_dataset("nateraw/parti-prompts", split="train")
    elif args.dataset == 'coco2017':
        dataset = load_dataset("phiyodr/coco2017")
        prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
    else:
        raise NotImplementedError

    # Fixing these sample prompts in the interest of reproducibility.

    if args.model == 'stable_diffusion_v1-5':
        from diffusers import DiffusionPipeline 
        pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(f"cuda:{args.gpu}")
    elif args.model == 'stable_diffusion_xl':
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(f"cuda:{args.gpu}")
    if args.scheduler == 'ddim':
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'dpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'euler':
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if args.adaptive_diffusion:
        import sys
        sys.path.append(Path(__file__).resolve().parent)

        if args.model == 'stable_diffusion_xl':
            from acceleration.sparse_pipeline import StableDiffusionXLPipeline as AdaptiveStableDiffusionXLPipeline
            adaptive_pipe = AdaptiveStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                threshold=args.threshold,
                                                                max_skip_steps=args.max_skip_steps,
                                                                torch_dtype=torch.float16).to("cuda:0")
        
        if args.model == 'stable_diffusion_v1-5':
            from acceleration.sparse_pipeline import StableDiffusionPipeline as AdaptiveStableDiffusionPipeline
            adaptive_pipe = AdaptiveStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                threshold=args.threshold,
                                                                max_skip_steps=args.max_skip_steps,
                                                                torch_dtype=torch.float16).to("cuda:0")
        adaptive_pipe.scheduler = pipe.scheduler
        pipe = adaptive_pipe

    image_list, prompt_list, config_list = [], [], []
    num_batch = len(prompts) // args.batch_size 
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    start_time = time.time()
    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)]
        set_random_seed(args.seed)
        
        pipe_output = pipe(sample_prompts, output_type="np", return_dict=True, num_inference_steps=args.steps)
        
        if args.adaptive_diffusion:
            skip_config = pipe.mask
            pipe.reset_cache()
            config_list.append(skip_config)

        images = pipe_output.images
        images_int = (images * 255).astype("uint8")
        torch_int_images = torch.from_numpy(images_int).permute(0, 3, 1, 2)
        image_list.append(torch_int_images)
        prompt_list += sample_prompts


    use_time = round(time.time() - start_time, 2)
    all_images = torch.cat(image_list, dim=0)

    if not os.path.exists(f"{args.dataset}_ckpt"):
        os.makedirs(f"{args.dataset}_ckpt")

    if args.original:
        torch.save({
            "images": all_images,
            "prompts": prompt_list,
        }, f"{args.dataset}_ckpt/images-original-{args.model}-{args.scheduler}-{args.steps}-bs-{args.batch_size}-time-{use_time}.pt")
    elif args.adaptive_diffusion:
        torch.save({
            "images": all_images,
            "prompts": prompt_list,
            "configs": config_list,
        }, f"{args.dataset}_ckpt/images-adaptive-{args.model}-{args.scheduler}-{args.steps}-bs-{args.batch_size}-max_skip_step-{args.max_skip_steps}-threshold-{args.threshold}-time-{use_time}.pt")
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='coco2017')
    parser.add_argument("--model", type=str, default='stable_diffusion_xl')
    parser.add_argument("--scheduler", type=str, default='ddim')

    parser.add_argument("--gpu", type=str, default='0')

    # For choosing baselines.
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--adaptive_diffusion", action="store_false")

    # Adaptive Skip Setup
    parser.add_argument("--max_skip_steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.01)

    # Sampling setup
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_random_seed(args.seed)
    
    main(args)