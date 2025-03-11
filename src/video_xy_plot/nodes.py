from inspect import cleandoc
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Import necessary ComfyUI components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "comfy"))
import comfy.model_management
import comfy.utils

class VideoXYPlot:
    """
    Creates an XY plot comparing different CFG and Shift values for videos.
    
    This node generates a grid of images where:
    - Each column represents a different CFG value
    - Each row represents a different Shift value
    - Each cell contains a generated image with those parameters
    - For videos, each frame will contain the complete grid
    
    This allows for easy comparison of how different CFG and Shift values affect the output,
    especially useful for finding optimal settings for SD3 models.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_max": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_steps": ("INT", {"default": 4, "min": 2, "max": 10, "step": 1}),
                "shift_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_max": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_steps": ("INT", {"default": 4, "min": 2, "max": 10, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_labels": (["enable", "disable"], {"default": "enable"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_xy_plot"
    CATEGORY = "video_xy_plot"

    def generate_xy_plot(self, model, vae, positive, negative, latent_image, 
                         cfg_min, cfg_max, cfg_steps, 
                         shift_min, shift_max, shift_steps,
                         steps, sampler_name, scheduler, seed, denoise, add_labels):
        # Import necessary modules here to avoid circular imports
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        
        # Generate arrays of CFG and Shift values
        cfg_values = np.linspace(cfg_min, cfg_max, cfg_steps)
        shift_values = np.linspace(shift_min, shift_max, shift_steps)
        
        # Check if we're dealing with a video (multiple frames)
        is_video = latent_image["samples"].shape[0] > 1
        num_frames = latent_image["samples"].shape[0] if is_video else 1
        
        # For video, we'll create a grid for each frame
        all_frame_grids = []
        
        # Process each frame
        for frame_idx in range(num_frames):
            # Create empty lists to store images for this frame
            images_rows = []
            
            # Extract the current frame's latent if it's a video
            if is_video:
                frame_latent = {"samples": latent_image["samples"][frame_idx:frame_idx+1].clone()}
                if "noise_mask" in latent_image:
                    frame_latent["noise_mask"] = latent_image["noise_mask"][frame_idx:frame_idx+1].clone() if latent_image["noise_mask"].shape[0] > 1 else latent_image["noise_mask"].clone()
            else:
                frame_latent = latent_image.copy()
            
            # Process each shift value
            for shift in shift_values:
                # Apply ModelSamplingSD3 to the model with current shift
                model_sd3 = ModelSamplingSD3().patch(model, shift)[0]
                
                # Process each CFG value for this shift
                images_row = []
                for cfg in cfg_values:
                    # Create a sampler
                    sampler = comfy.samplers.KSampler(model_sd3, steps, comfy.model_management.get_torch_device(), 
                                                     sampler=sampler_name, scheduler=scheduler, denoise=denoise)
                    
                    # Generate noise based on the seed
                    noise = torch.randn(frame_latent["samples"].shape, 
                                       dtype=frame_latent["samples"].dtype, 
                                       device=frame_latent["samples"].device, 
                                       generator=torch.manual_seed(seed + frame_idx if is_video else seed))
                    
                    # Sample with the current CFG value
                    samples = sampler.sample(noise, positive, negative, cfg, latent_image=frame_latent["samples"])
                    
                    # Decode the latent using VAE
                    image = vae.decode(samples)
                    
                    # Add to the row
                    images_row.append(image)
                
                # Concatenate images horizontally for this row (CFG values)
                row_tensor = torch.cat(images_row, dim=2)
                images_rows.append(row_tensor)
            
            # Concatenate all rows vertically (Shift values)
            frame_grid = torch.cat(images_rows, dim=1)
            
            # Add labels if enabled (only need to do this once for videos)
            if add_labels == "enable" and (frame_idx == 0 or not is_video):
                # Convert to numpy for PIL
                grid_np = frame_grid.cpu().numpy()
                # Normalize to 0-255 range and convert to uint8
                grid_np = (grid_np * 255).astype(np.uint8)
                
                # Convert to PIL Image for drawing
                pil_image = Image.fromarray(grid_np[0].transpose(1, 2, 0))
                draw = ImageDraw.Draw(pil_image)
                
                # Try to load a font, use default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                
                # Get image dimensions
                width, height = pil_image.size
                cell_width = width // cfg_steps
                cell_height = height // shift_steps
                
                # Add CFG labels (top)
                for i, cfg in enumerate(cfg_values):
                    x = i * cell_width + cell_width // 2
                    draw.text((x, 10), f"CFG: {cfg:.2f}", fill=(255, 255, 255), font=font, anchor="mt")
                
                # Add Shift labels (left)
                for i, shift in enumerate(shift_values):
                    y = i * cell_height + cell_height // 2
                    draw.text((10, y), f"Shift: {shift:.2f}", fill=(255, 255, 255), font=font, anchor="lm")
                
                # Convert back to tensor
                grid_np = np.array(pil_image).transpose(2, 0, 1)
                grid_np = grid_np.astype(np.float32) / 255.0
                frame_grid = torch.from_numpy(grid_np).unsqueeze(0)
            
            # Add this frame's grid to our collection
            all_frame_grids.append(frame_grid)
        
        # If it's a video, stack all frames together
        if is_video:
            final_grid = torch.cat(all_frame_grids, dim=0)
        else:
            final_grid = all_frame_grids[0]
        
        return (final_grid,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoXYPlot": VideoXYPlot
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoXYPlot": "Video XY Plot (CFG-Shift)"
}
