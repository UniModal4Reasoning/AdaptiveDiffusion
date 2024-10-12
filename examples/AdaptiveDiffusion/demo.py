import time
import numpy as np
import random
from pathlib import Path
import torch
import gradio as gr
import threading
import sys
sys.path.append(str(Path(__file__).resolve().parent))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_original_image(pipe, prompt, steps, result_dict):
    start_time = time.time()
    original_image = pipe(prompt, num_inference_steps=steps, output_type="pil").images[0]
    original_time = time.time() - start_time
    result_dict['original_image'] = original_image
    result_dict['original_time'] = original_time

def generate_adaptive_image(pipe, prompt, steps, result_dict):
    start_time = time.time()
    adaptive_image = pipe(prompt, num_inference_steps=steps, output_type="pil").images[0]
    adaptive_time = time.time() - start_time
    pipe.reset_cache()
    result_dict['adaptive_image'] = adaptive_image
    result_dict['adaptive_time'] = adaptive_time

def create_pipelines(model_name, threshold, max_skip_steps):
    if model_name == 'stable_diffusion_v1-5':
        from diffusers import DiffusionPipeline 
        pipe_original = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda:0")
    elif model_name == 'stable_diffusion_xl':
        from diffusers import StableDiffusionXLPipeline
        pipe_original = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True).to("cuda:0")
    else:
        return None, None, "Model not supported."

    if model_name == 'stable_diffusion_xl':
        from acceleration.sparse_pipeline import StableDiffusionXLPipeline as AdaptiveStableDiffusionXLPipeline
        pipe_adaptive = AdaptiveStableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            threshold=threshold,
            max_skip_steps=max_skip_steps,
            torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda:2")
    elif model_name == 'stable_diffusion_v1-5':
        from acceleration.sparse_pipeline import StableDiffusionPipeline as AdaptiveStableDiffusionPipeline
        pipe_adaptive = AdaptiveStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            threshold=threshold,
            max_skip_steps=max_skip_steps,
            torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda:2")
    else:
        return None, None, "Adaptive diffusion model not supported."
    
    return pipe_original, pipe_adaptive, None

def create_schedulers(pipe_original, pipe_adaptive, scheduler_name):
    if scheduler_name == 'ddim':
        from diffusers import DDIMScheduler
        pipe_original.scheduler = DDIMScheduler.from_config(pipe_original.scheduler.config)
        pipe_adaptive.scheduler = DDIMScheduler.from_config(pipe_adaptive.scheduler.config)
    elif scheduler_name == 'dpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe_original.scheduler = DPMSolverMultistepScheduler.from_config(pipe_original.scheduler.config)
        pipe_adaptive.scheduler = DPMSolverMultistepScheduler.from_config(pipe_adaptive.scheduler.config)
    elif scheduler_name == 'euler':
        from diffusers import EulerDiscreteScheduler
        pipe_original.scheduler = EulerDiscreteScheduler.from_config(pipe_original.scheduler.config)
        pipe_adaptive.scheduler = EulerDiscreteScheduler.from_config(pipe_adaptive.scheduler.config)
    else:
        return None, None, "Scheduler not supported."

    return pipe_original, pipe_adaptive, None

def main():
    model_options = ['stable_diffusion_v1-5', 'stable_diffusion_xl']
    scheduler_options = ['ddim', 'dpm', 'euler']

    default_model_name = 'stable_diffusion_xl'
    default_scheduler_name = 'ddim'
    default_threshold = 0.01
    default_max_skip_steps = 4

    pipe_original, pipe_adaptive, error = create_pipelines(
        default_model_name, default_threshold, default_max_skip_steps
    )
    pipe_original, pipe_adaptive, error = create_schedulers(
        pipe_original, pipe_adaptive, default_scheduler_name
    )
    if error:
        print(error)
        return

    with gr.Blocks() as demo:
        gr.Markdown("## Demo of AdaptiveDiffusion vs. Full-step Diffusion")
        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(model_options, value=default_model_name, label="Select Model")
                scheduler_select = gr.Dropdown(scheduler_options, value=default_scheduler_name, label="Select Scheduler")
            with gr.Column():
                seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                steps_input = gr.Number(value=50, label="Number of Steps", precision=0)
            with gr.Column():
                threshold_input = gr.Number(value=default_threshold, label="Threshold", precision=3)
                max_skip_steps_input = gr.Number(value=default_max_skip_steps, label="Max Skip Steps", precision=0)
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                generate_button = gr.Button("Generate Images")

        with gr.Row():
            with gr.Column():
                original_image_output = gr.Image(label="Full-step Diffusion Output")
                original_time_output = gr.Textbox(label="Full-step Diffusion Time (s)", interactive=False)
            with gr.Column():
                adaptive_image_output = gr.Image(label="AdaptiveDiffusion Output")
                adaptive_time_output = gr.Textbox(label="AdaptiveDiffusion Time (s)", interactive=False)
                error_output = gr.Textbox(label="Error Messages", visible=False)

        state = gr.State({
            'pipe_original': pipe_original,
            'pipe_adaptive': pipe_adaptive,
            'model_name': default_model_name,
            'scheduler_name': default_scheduler_name,
            'threshold': default_threshold,
            'max_skip_steps': default_max_skip_steps
        })

        def update_interface(prompt, model_name, scheduler_name, seed, steps, threshold, max_skip_steps, state):
            if model_name != state['model_name']:
                pipe_original, pipe_adaptive, error = create_pipelines(
                    model_name, threshold, max_skip_steps
                )
                if error:
                    yield gr.update(value=error, visible=True), None, None, None, None, state
                    return
                state['pipe_original'] = pipe_original
                state['pipe_adaptive'] = pipe_adaptive
                state['model_name'] = model_name
                state['scheduler_name'] = scheduler_name
                state['threshold'] = threshold
                state['max_skip_steps'] = max_skip_steps
            elif scheduler_name != state['scheduler_name']:
                pipe_original = state['pipe_original']
                pipe_adaptive = state['pipe_adaptive']
                pipe_original, pipe_adaptive, error = create_schedulers(
                    pipe_original, pipe_adaptive, scheduler_name
                )
                if error:
                    yield gr.update(value=error, visible=True), None, None, None, None, state
                    return
                pipe_adaptive.threshold = threshold
                pipe_adaptive.max_skip_steps = max_skip_steps
                state['pipe_original'] = pipe_original
                state['pipe_adaptive'] = pipe_adaptive
                state['scheduler_name'] = scheduler_name
                state['threshold'] = threshold
                state['max_skip_steps'] = max_skip_steps
            else:
                pipe_original = state['pipe_original']
                pipe_adaptive = state['pipe_adaptive']
                pipe_adaptive.threshold = threshold
                pipe_adaptive.max_skip_steps = max_skip_steps
                state['threshold'] = threshold
                state['max_skip_steps'] = max_skip_steps

            set_random_seed(int(seed))

            error_output = gr.update(visible=False)
            original_image_output = None
            original_time_output = None
            adaptive_image_output = None
            adaptive_time_output = None

            result_dict = {}
            threads = []
            thread_original = threading.Thread(target=generate_original_image, args=(pipe_original, prompt, int(steps), result_dict))
            thread_adaptive = threading.Thread(target=generate_adaptive_image, args=(pipe_adaptive, prompt, int(steps), result_dict))
            threads = [thread_original, thread_adaptive]
            
            for t in threads:
                t.start()

            yield (
                error_output,
                original_image_output,
                original_time_output,
                adaptive_image_output,
                adaptive_time_output,
                state
            )

            while any(t.is_alive() for t in threads):
                time.sleep(0.1)
                if 'original_image' in result_dict and original_image_output is None:
                    original_image_output = result_dict['original_image']
                    original_time_output = f"{result_dict['original_time']:.2f} seconds"
                    yield (
                        error_output,
                        original_image_output,
                        original_time_output,
                        adaptive_image_output,
                        adaptive_time_output,
                        state
                    )
                if 'adaptive_image' in result_dict and adaptive_image_output is None:
                    adaptive_image_output = result_dict['adaptive_image']
                    adaptive_time_output = f"{result_dict['adaptive_time']:.2f} seconds"
                    yield (
                        error_output,
                        original_image_output,
                        original_time_output,
                        adaptive_image_output,
                        adaptive_time_output,
                        state
                    )

            if 'original_image' in result_dict:
                original_image_output = result_dict['original_image']
                original_time_output = f"{result_dict['original_time']:.2f} seconds"
            if 'adaptive_image' in result_dict:
                adaptive_image_output = result_dict['adaptive_image']
                adaptive_time_output = f"{result_dict['adaptive_time']:.2f} seconds"

            yield (
                error_output,
                original_image_output,
                original_time_output,
                adaptive_image_output,
                adaptive_time_output,
                state
            )

        generate_button.click(
            update_interface,
            inputs=[
                prompt_input,
                model_select,
                scheduler_select,
                seed_input,
                steps_input,
                threshold_input,
                max_skip_steps_input,
                state
            ],
            outputs=[
                error_output,
                original_image_output,
                original_time_output,
                adaptive_image_output,
                adaptive_time_output,
                state
            ]
        )
    demo.launch(share=True)

if __name__ == '__main__':
    main()