#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
DPG-Bench image generation with 4 images per prompt (2x2 grid).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from PIL import Image


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_rank() -> int:
    """Get the current process rank."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def load_backbone(backbone: str, backbone_cfg: Dict[str, Any]) -> Any:
    """Load backbone model on rank 0 only, others wait at barrier."""
    rank = get_rank()
    world_size = get_world_size()
    
    if rank == 0:
        print(f"Rank 0: Initializing {backbone} backbone...")
        from umm.inference.pipeline import InferencePipeline
        pipeline = InferencePipeline(backbone, backbone_cfg)
        print(f"Rank 0: Backbone loaded successfully")
    else:
        pipeline = None
        print(f"Rank {rank}: Waiting for rank 0 to load model...")
    
    # Synchronize all ranks after loading
    dist.barrier()
    
    return pipeline


def grid_images(images: List[Image.Image], grid_size: tuple = (2, 2)) -> Image.Image:
    """Grid multiple images into a single image."""
    if len(images) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} images, got {len(images)}")
    
    # Ensure all images are the same size
    img_width, img_height = images[0].size
    
    # Create grid
    grid_width = img_width * grid_size[1]
    grid_height = img_height * grid_size[0]
    grid = Image.new("RGB", (grid_width, grid_height))
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid


def generate_images_distributed(
    backbone: str,
    backbone_cfg: Dict[str, Any],
    prompts_dir: str,
    output_dir: str,
    num_images_per_prompt: int = 4,
    cfg_text_scale: float = 4.0,
    cfg_interval: List[float] = None,
    cfg_renorm_min: float = 0.0,
    timestep_shift: float = 4.0,
    num_timesteps: int = 50,
) -> None:
    """Generate images for DPG-Bench with gridding."""
    
    if cfg_interval is None:
        cfg_interval = [0, 1.0]
    
    # Initialize distributed environment
    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    
    # Get all prompt files
    prompt_files = sorted([f for f in Path(prompts_dir).glob("*.txt")])
    total_prompts = len(prompt_files)
    
    print(f"GPU {rank}: Total prompts to process: {total_prompts}")
    
    # Load backbone on rank 0 only (others wait at barrier)
    pipeline = load_backbone(backbone, backbone_cfg)
    
    # Only rank 0 generates images
    if rank == 0:
        backbone_obj = pipeline.backbone
        
        print(f"Rank 0: Starting image generation for {total_prompts} prompts")
        
        # Generate images
        for idx, prompt_file in enumerate(prompt_files):
            prompt_name = prompt_file.stem  # filename without .txt
            
            # Read prompt
            with open(prompt_file, "r") as f:
                prompt = f.read().strip()
            
            # Output image path (same name as prompt file, but .png)
            output_path = os.path.join(output_dir, f"{prompt_name}.png")
            
            print(f"Rank 0 processing prompt {idx + 1}/{total_prompts}: '{prompt_name}' ('{prompt}')")
            
            # Skip if already generated
            if os.path.exists(output_path):
                print(f"Rank 0 skipping generation for prompt: {prompt_name}")
                continue
            
            try:
                # Prepare generation config
                gen_cfg = {
                    "num_timesteps": num_timesteps,
                    "cfg_text_scale": cfg_text_scale,
                    "cfg_interval": cfg_interval,
                    "cfg_renorm_min": cfg_renorm_min,
                    "timestep_shift": timestep_shift,
                }
                
                # Generate multiple images per prompt
                generated_images = []
                for img_idx in range(num_images_per_prompt):
                    print(f"Rank 0 generating image {img_idx + 1}/{num_images_per_prompt} for '{prompt_name}'")
                    
                    # Generate single image
                    if hasattr(backbone_obj, "generation"):
                        result = backbone_obj.generation(
                            prompt=prompt,
                            output_path=None,  # Don't save individual images
                            generation_cfg=gen_cfg,
                        )
                        # Extract image from result dict
                        if isinstance(result, dict) and "image" in result:
                            img = result["image"]
                        else:
                            img = result
                    elif hasattr(backbone_obj, "generate"):
                        batch = {"prompt": prompt}
                        result = backbone_obj.generate(batch=batch, gen_cfg=gen_cfg)
                        if isinstance(result, dict) and "image" in result:
                            img = result["image"]
                        else:
                            img = result
                    else:
                        raise NotImplementedError(f"Backbone {backbone} does not implement generation")
                    
                    # Convert to PIL Image if necessary
                    if not isinstance(img, Image.Image):
                        if isinstance(img, torch.Tensor):
                            # Convert tensor to PIL Image
                            if img.ndim == 3:
                                if img.shape[0] == 3:  # CHW format
                                    img = img.permute(1, 2, 0)
                                img = (img * 255).clamp(0, 255).byte().cpu().numpy()
                            img = Image.fromarray(img)
                        else:
                            raise ValueError(f"Unexpected result type: {type(img)}")
                    
                    generated_images.append(img)
                
                # Grid images into 2x2
                if num_images_per_prompt == 4:
                    grid_img = grid_images(generated_images, grid_size=(2, 2))
                else:
                    # For other numbers, just use the first image or create a different grid
                    print(f"Rank 0 warning: num_images_per_prompt={num_images_per_prompt}, expected 4")
                    grid_img = generated_images[0] if generated_images else None
                
                # Save gridded image
                if grid_img:
                    grid_img.save(output_path)
                    print(f"Rank 0 saved gridded image: {output_path}")
                
            except Exception as e:
                print(f"Rank 0 error processing prompt {prompt_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Rank 0 has completed all image generation tasks")
    else:
        print(f"Rank {rank}: Waiting for generation to complete...")
    
    # All ranks wait at barrier
    dist.barrier()
    
    # Cleanup
    if rank == 0:
        print("Rank 0: Destroying process group")
        dist.destroy_process_group()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DPG-Bench Image Generation with 4 images per prompt (2x2 grid)"
    )
    
    # Required arguments
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Backbone model to use (e.g., bagel, janus_pro)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="./eval/generation/dpg_bench/prompts",
        help="Directory containing prompt .txt files"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images per prompt (default: 4)"
    )
    
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=50,
        help="Number of diffusion timesteps (default: 50)"
    )
    
    parser.add_argument(
        "--cfg_text_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance text scale (default: 4.0)"
    )
    
    parser.add_argument(
        "--cfg_interval",
        type=str,
        default="0,1.0",
        help="CFG interval as comma-separated floats (default: 0,1.0)"
    )
    
    parser.add_argument(
        "--cfg_renorm_min",
        type=float,
        default=0.0,
        help="CFG renormalization minimum (default: 0.0)"
    )
    
    parser.add_argument(
        "--timestep_shift",
        type=float,
        default=4.0,
        help="Timestep shift for generation (default: 4.0)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse CFG interval
    cfg_interval = [float(x) for x in args.cfg_interval.split(",")]
    
    # Prepare backbone config
    backbone_cfg = {}
    if args.model_path:
        backbone_cfg["model_path"] = args.model_path
    
    # Run generation
    print(f"Starting DPG-Bench generation with {args.backbone} backbone")
    print(f"Output directory: {args.output_dir}")
    print(f"Prompts directory: {args.prompts_dir}")
    print(f"Images per prompt: {args.num_images_per_prompt}")
    
    generate_images_distributed(
        backbone=args.backbone,
        backbone_cfg=backbone_cfg,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images_per_prompt,
        cfg_text_scale=args.cfg_text_scale,
        cfg_interval=cfg_interval,
        cfg_renorm_min=args.cfg_renorm_min,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
    )
    
    print("Generation completed successfully")


if __name__ == "__main__":
    main()
