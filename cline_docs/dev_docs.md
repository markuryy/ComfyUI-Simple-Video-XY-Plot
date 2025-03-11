# Video XY Plot for ComfyUI - Developer Documentation

## Product Context

### Purpose
The Video XY Plot node for ComfyUI is designed to help users find optimal parameters for SD3 models by providing a visual comparison of different CFG and Shift values. This is particularly useful for fine-tuning generation parameters to achieve the best results.

### Problems Solved
1. **Parameter Optimization**: Finding the right combination of CFG and Shift values for SD3 models is often a trial-and-error process. This node automates the generation of multiple variations, saving time and effort.
2. **Visual Comparison**: By arranging the results in a grid, users can easily compare the effects of different parameter combinations side by side.
3. **Video Support**: The node handles both single images and videos, allowing users to see how parameter changes affect animation consistency.

### How It Works
1. The node takes a model and applies ModelSamplingSD3 with different shift values.
2. For each shift value, it runs KSampler with different CFG values.
3. It decodes the latents using VAE.
4. It arranges the results in a grid where:
   - Columns represent different CFG values
   - Rows represent different Shift values
   - Each cell contains an image with those parameters
5. It adds labels to identify the CFG and Shift values.
6. For videos, each frame contains a complete grid, allowing users to see how parameter changes affect the entire animation.

## Active Context

### Current Work
We have successfully implemented the Video XY Plot node for ComfyUI that creates a grid comparing different CFG and Shift values for SD3 models. This node is designed to help users find optimal parameters for their models by providing a visual comparison of different combinations.

### Recent Changes
1. **Initial Implementation**: Created the VideoXYPlot class that:
   - Takes a model, VAE, and other parameters
   - Creates an XY plot comparing different CFG and Shift values
   - Handles both single images and videos
   - Adds labels to identify the CFG and Shift values

2. **Video Support**: Added support for processing video frames, where each frame contains a complete grid of parameter combinations.

### Next Steps
1. **Testing**: Test the node with different models, especially SD3 models, to ensure it works as expected.
2. **Performance Optimization**: Optimize the node for better performance, especially for videos with many frames.
3. **Enhanced Labeling**: Improve the labeling system to make it more customizable and visually appealing.
4. **Additional Parameters**: Consider adding support for comparing other parameters beyond CFG and Shift.

## System Patterns

### Architecture Overview
The Video XY Plot node is built as a custom node for ComfyUI, following the established patterns and conventions of the ComfyUI framework. It integrates with existing components like ModelSamplingSD3, KSampler, and VAEDecode to create a comprehensive parameter comparison tool.

### Key Technical Decisions

1. **Node Structure**
   - Class-Based Implementation: The node is implemented as a Python class (VideoXYPlot) that follows the ComfyUI node structure.
   - Input/Output Definition: Clear definition of input types and return types using ComfyUI's type system.
   - Function-Based Processing: The main processing logic is contained in the generate_xy_plot function.

2. **Parameter Handling**
   - Range-Based Parameters: Instead of individual values, the node accepts ranges (min, max, steps) for both CFG and Shift parameters.
   - Linear Spacing: Uses numpy's linspace to generate evenly spaced values within the specified ranges.

3. **Image Processing**
   - Grid Construction: Images are concatenated horizontally for each row (CFG values) and vertically for each column (Shift values).
   - Label Addition: Uses PIL to add text labels to the grid for better readability.
   - Video Support: Handles video frames by creating a grid for each frame and stacking them together.

4. **Integration with ComfyUI**
   - Model Modification: Uses ModelSamplingSD3 to modify the model with different shift values.
   - Sampling: Uses KSampler to generate images with different CFG values.
   - Decoding: Uses VAE to decode latents to images.

## Technical Context

### Technologies Used
- **Python**: The primary programming language used for implementation.
- **PyTorch**: Used for tensor operations and model handling.
- **NumPy**: Used for numerical operations and array handling.
- **PIL (Pillow)**: Used for image processing and text rendering.
- **ComfyUI Framework**: The underlying framework for node creation and integration.

### Key Components
1. **ModelSamplingSD3**: From comfy_extras/nodes_model_advanced.py, used to modify the model with different shift values.
2. **KSampler**: From comfy/samplers.py, used to generate images with different CFG values.
3. **VAEDecode**: From nodes.py, used to decode latents to images.

### File Structure
- **src/video_xy_plot/nodes.py**: Contains the VideoXYPlot class implementation.
- **src/video_xy_plot/__init__.py**: Imports and exports the node class mappings.
- **__init__.py**: Main package initialization file that imports from the src module.

## Progress Status

### Completed
- ✅ Basic node structure and implementation
- ✅ Parameter handling for CFG and Shift values
- ✅ Grid construction for visual comparison
- ✅ Label addition for better readability
- ✅ Video support for frame-by-frame comparison
- ✅ Integration with ComfyUI framework

### In Progress
- 🔄 Testing with different models and parameters
- 🔄 Documentation and examples

### To Do
- ⬜ Performance optimization for videos
- ⬜ Enhanced labeling system
- ⬜ Support for additional parameters
- ⬜ User interface improvements

## Reference Files

### Core Components
1. **ModelSamplingSD3**
   - File: `../../comfy_extras/nodes_model_advanced.py`
   - Description: Modifies a model with a "shift" parameter
   - Key method: `patch(self, model, shift, multiplier=1000)`

2. **KSampler**
   - File: `../../nodes.py`
   - Description: Handles the sampling process with parameters like CFG
   - Key method: `sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0)`

3. **VAEDecode**
   - File: `../../nodes.py`
   - Description: Decodes latent images into pixel space
   - Key method: `decode(self, vae, samples)`

### Supporting Components
4. **Sample Module**
   - File: `../../comfy/sample.py`
   - Description: Contains helper functions for sampling

5. **Samplers Module**
   - File: `../../comfy/samplers.py`
   - Description: Contains sampler implementations

6. **Model Sampling Module**
   - File: `../../comfy/model_sampling.py`
   - Description: Contains model sampling implementations including the base classes that ModelSamplingSD3 inherits from
