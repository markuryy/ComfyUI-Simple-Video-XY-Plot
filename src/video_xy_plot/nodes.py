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
import comfy.sample

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
        
        # Add debug logging
        print(f"Starting VideoXYPlot with cfg_steps={cfg_steps}, shift_steps={shift_steps}")
        
        # Generate arrays of CFG and Shift values
        cfg_values = np.linspace(cfg_min, cfg_max, cfg_steps)
        shift_values = np.linspace(shift_min, shift_max, shift_steps)
        
        # Check if we're dealing with a video (multiple frames)
        # For ComfyUI, video detection is based on the batch dimension (index 0)
        is_video = latent_image["samples"].shape[0] > 1
        num_frames = latent_image["samples"].shape[0] if is_video else 1
        print(f"Input latent shape: {latent_image['samples'].shape}, is_video: {is_video}, num_frames: {num_frames}")
        
        # Create a list to store all generated videos/images for each cfg/shift combination
        all_videos = []
        
        # Process each shift value
        for shift_idx, shift in enumerate(shift_values):
            # Apply ModelSamplingSD3 to the model with current shift
            model_sd3 = ModelSamplingSD3().patch(model, shift)[0]
            
            # Process each CFG value for this shift
            row_videos = []
            for cfg_idx, cfg in enumerate(cfg_values):
                print(f"Processing shift={shift:.2f}, cfg={cfg:.2f}")
                
                # Create a list to store all frames for this video
                video_frames = []
                
                # Process each frame with the same cfg/shift combination
                for frame_idx in range(num_frames):
                    # Extract the current frame's latent if it's a video
                    if is_video:
                        frame_latent = {"samples": latent_image["samples"][frame_idx:frame_idx+1].clone()}
                        if "noise_mask" in latent_image:
                            frame_latent["noise_mask"] = latent_image["noise_mask"][frame_idx:frame_idx+1].clone() if latent_image["noise_mask"].shape[0] > 1 else latent_image["noise_mask"].clone()
                    else:
                        frame_latent = latent_image.copy()
                    
                    # Generate noise based on the seed
                    noise = torch.randn(frame_latent["samples"].shape, 
                                       dtype=frame_latent["samples"].dtype, 
                                       device=frame_latent["samples"].device, 
                                       generator=torch.manual_seed(seed + frame_idx if is_video else seed))
                    
                    # Sample using comfy.sample.sample which handles the preview
                    samples = comfy.sample.sample(model_sd3, noise, steps, cfg, sampler_name, scheduler,
                                                 positive, negative, frame_latent["samples"],
                                                 denoise=denoise)
                    
                    # Decode the latent using VAE
                    image = vae.decode(samples)
                    print(f"  Frame {frame_idx} decoded shape: {image.shape}")
                    
                    # Ensure image has the right dimensions (handle 5D tensors from VAE)
                    if len(image.shape) == 5:  # [batch, frames, channel, height, width]
                        # Reshape to [frames, channel, height, width]
                        image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
                        print(f"  Reshaped 5D tensor to: {image.shape}")
                    
                    # Add to the frames for this video
                    video_frames.append(image)
                
                # Stack all frames for this cfg/shift combination into a video
                if is_video:
                    video = torch.cat(video_frames, dim=0)
                else:
                    video = video_frames[0]
                
                print(f"  Video shape after stacking frames: {video.shape}")
                
                # Add to the row
                row_videos.append(video)
            
            # Add this row of videos to our collection
            all_videos.append(row_videos)
        
        print(f"All videos collected. Grid size: {len(all_videos)}x{len(all_videos[0])}")
        
        # Now arrange the videos in a grid
        # First, concatenate each row horizontally (CFG values)
        rows = []
        for row_idx, row in enumerate(all_videos):
            print(f"Processing row {row_idx} for grid")
            # For each video in the row, we need to handle all frames
            if is_video:
                # We need to concatenate frame by frame
                row_frames = []
                for frame_idx in range(num_frames):
                    # Extract the current frame from each video in the row
                    frame_row = [video[frame_idx:frame_idx+1] for video in row]
                    # Log shapes for debugging
                    frame_shapes = [f.shape for f in frame_row]
                    print(f"  Frame {frame_idx} shapes in row: {frame_shapes}")
                    # Concatenate horizontally
                    frame_row = torch.cat(frame_row, dim=3)  # Concatenate along width (dim=3)
                    print(f"  Concatenated frame row shape: {frame_row.shape}")
                    row_frames.append(frame_row)
                # Stack all frames for this row
                row_tensor = torch.cat(row_frames, dim=0)
                print(f"  Row tensor shape after stacking frames: {row_tensor.shape}")
            else:
                # For images, just concatenate horizontally
                row_tensor = torch.cat(row, dim=2)  # Concatenate along width (dim=2)
                print(f"  Row tensor shape (image mode): {row_tensor.shape}")
            rows.append(row_tensor)
        
        # Then, concatenate all rows vertically (Shift values)
        if is_video:
            # For videos, we need to concatenate frame by frame
            final_frames = []
            for frame_idx in range(num_frames):
                # Extract the current frame from each row
                frame_rows = [row[frame_idx:frame_idx+1] for row in rows]
                # Log shapes for debugging
                frame_shapes = [f.shape for f in frame_rows]
                print(f"  Frame {frame_idx} shapes from rows: {frame_shapes}")
                # Concatenate vertically
                frame_grid = torch.cat(frame_rows, dim=2)  # Concatenate along height (dim=2)
                print(f"  Frame grid shape after vertical concat: {frame_grid.shape}")
                final_frames.append(frame_grid)
            # Stack all frames
            final_grid = torch.cat(final_frames, dim=0)
            print(f"Final grid shape (video mode): {final_grid.shape}")
        else:
            # For images, just concatenate vertically
            final_grid = torch.cat(rows, dim=1)  # Concatenate along height (dim=1)
            print(f"Final grid shape (image mode): {final_grid.shape}")
        
        # Check if the final grid has multiple frames, regardless of input
        has_multiple_frames = final_grid.shape[0] > 1
        print(f"Final grid has multiple frames: {has_multiple_frames}, shape: {final_grid.shape}")
        
        # Add labels if enabled
        if add_labels == "enable":
            # We'll add labels to the first frame only
            first_frame = final_grid[0:1] if has_multiple_frames else final_grid
            print(f"First frame shape for labeling: {first_frame.shape}")
            
            # Convert to numpy for PIL
            grid_np = first_frame.cpu().numpy()
            print(f"Numpy array shape for PIL conversion: {grid_np.shape}")
            
            # Handle different tensor shapes
            if len(grid_np.shape) == 5:  # [batch, frames, channel, height, width]
                print("Detected 5D tensor, reshaping for PIL conversion")
                # Reshape to 4D by combining batch and frames
                grid_np = grid_np.reshape(-1, grid_np.shape[-3], grid_np.shape[-2], grid_np.shape[-1])
                print(f"Reshaped to: {grid_np.shape}")
            
            # Normalize to 0-255 range and convert to uint8
            grid_np = (grid_np * 255).astype(np.uint8)
            
            # Create PIL image based on tensor shape
            try:
                # Based on the error log, we need to handle the specific shape (5, 1664, 960, 3)
                # This is [frames, height, width, channel] format
                if len(grid_np.shape) == 4 and grid_np.shape[-1] == 3:  # [frames/batch, height, width, channel]
                    # This is already in the right format for PIL
                    pil_image = Image.fromarray(grid_np[0])
                    print(f"Created PIL image with size: {pil_image.size}")
                elif len(grid_np.shape) == 4:  # [batch, channel, height, width]
                    # Standard format: [batch, channel, height, width] -> [height, width, channel]
                    pil_image = Image.fromarray(grid_np[0].transpose(1, 2, 0))
                    print(f"Created PIL image with size: {pil_image.size}")
                elif len(grid_np.shape) == 3 and grid_np.shape[-1] == 3:  # [height, width, channel]
                    # Already in the right format
                    pil_image = Image.fromarray(grid_np)
                    print(f"Created PIL image with size: {pil_image.size}")
                elif len(grid_np.shape) == 3:  # [channel, height, width]
                    # No batch dimension: [channel, height, width] -> [height, width, channel]
                    pil_image = Image.fromarray(grid_np.transpose(1, 2, 0))
                    print(f"Created PIL image with size: {pil_image.size}")
                else:
                    raise ValueError(f"Unexpected tensor shape: {grid_np.shape}")
            except Exception as e:
                print(f"Error creating PIL image: {e}")
                print(f"Tensor shape: {grid_np.shape}")
                print(f"Data type: {grid_np.dtype}")
                print(f"Min/max values: {grid_np.min()}, {grid_np.max()}")
                
                # Create a blank image with dimensions from the tensor if possible
                if len(grid_np.shape) >= 3:
                    if len(grid_np.shape) == 4:
                        if grid_np.shape[-1] == 3:  # [frames/batch, height, width, channel]
                            h, w = grid_np.shape[1], grid_np.shape[2]
                        else:  # [batch, channel, height, width]
                            h, w = grid_np.shape[2], grid_np.shape[3]
                    else:  # [channel, height, width] or [height, width, channel]
                        if grid_np.shape[-1] == 3:  # [height, width, channel]
                            h, w = grid_np.shape[0], grid_np.shape[1]
                        else:  # [channel, height, width]
                            h, w = grid_np.shape[1], grid_np.shape[2]
                    pil_image = Image.new('RGB', (w, h), color=(0, 0, 0))
                    print(f"Created blank image with size: {pil_image.size}")
                else:
                    # Fallback to a default size
                    pil_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
                    print("Created default 512x512 blank image")
            
            # Create drawing context
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
            pil_array = np.array(pil_image)
            print(f"PIL array shape after drawing: {pil_array.shape}")
            # Keep as [height, width, channel] format which is what ComfyUI expects
            pil_array = pil_array.astype(np.float32) / 255.0
            labeled_frame = torch.from_numpy(pil_array).unsqueeze(0)
            print(f"Labeled frame tensor shape: {labeled_frame.shape}")
            
            # Replace the first frame with the labeled one
            if has_multiple_frames:
                print(f"Replacing first frame in video. Final grid shape before: {final_grid.shape}")
                final_grid = torch.cat([labeled_frame, final_grid[1:]], dim=0)
                print(f"Final grid shape after label replacement: {final_grid.shape}")
            else:
                final_grid = labeled_frame
                print(f"Final image shape after labeling: {final_grid.shape}")
        
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
