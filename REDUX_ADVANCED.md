# Advanced Redux with Patch Downsampling

This branch implements advanced Redux functionality using true patch downsampling, following the ComfyUI approach for superior control over how reference images influence generation.

## Core Technology: Patch Downsampling

### What is Patch Downsampling?

Instead of simply scaling final embeddings, this implementation controls the **amount of visual information** processed from reference images by manipulating the token structure before it reaches the text encoder.

**Technical Process:**
1. SigLIP encoder produces 27×27 = 729 spatial patches from the reference image
2. Redux encoder converts these to text-space tokens
3. **Downsampling step**: Groups patches into blocks and averages them
4. Resulting tokens have reduced spatial resolution but maintain semantic content

### Why This Works Better

- **ComfyUI Proven**: This is the exact method used in ComfyUI's successful Redux control
- **Information Control**: Controls visual information at the source, not just the output
- **Semantic Preservation**: Maintains important visual concepts while reducing dominance
- **Natural Balance**: Creates more natural prompt/image balance

## Redux Modes

### 🟡 Lowest Mode
- **Downsampling**: Maximum (5x) - Minimal image influence
- **Effect**: Prompt-driven generation with subtle image hints
- **Use Case**: When you want only loose inspiration from the reference image
- **Example**: Reference image provides general mood/color palette

### 🟠 Low Mode
- **Downsampling**: High (4x) - Low image influence
- **Effect**: Light image guidance with strong prompt creativity
- **Use Case**: Subtle style hints while maintaining prompt direction
- **Example**: Color palette or lighting inspiration

### 🟢 Medium Mode (Default)
- **Downsampling**: Moderate (3x) - ComfyUI default
- **Effect**: Balanced influence between image and prompt
- **Use Case**: Most scenarios where you want clear but not overwhelming image influence
- **Example**: Style transfer with creative freedom

### 🟠 High Mode
- **Downsampling**: Light (2x) - High image influence
- **Effect**: Strong image guidance with some prompt flexibility
- **Use Case**: When you want strong adherence but controlled
- **Example**: Detailed style transfer with some creative freedom

### 🔴 Highest Mode
- **Downsampling**: Minimal (1x) - Maximum image influence
- **Effect**: Image-focused generation with prompt refinement
- **Use Case**: Maximum fidelity to reference image
- **Example**: Precise style or composition transfer

## Interpolation Modes (ComfyUI Compatible)

### 🔹 Area Mode (Default)
- **Method**: Weighted average of source pixels
- **Quality**: Smooth and balanced results
- **Best For**: General use, natural photos, balanced detail preservation
- **ComfyUI**: Default method, most tested

### 🔸 Bicubic Mode
- **Method**: Cubic polynomial interpolation
- **Quality**: Very smooth, may introduce slight blur
- **Best For**: Artistic images, when you want maximum smoothness
- **Note**: Can sometimes "invent" details that weren't in the original

### 🔺 Nearest Mode
- **Method**: Nearest neighbor sampling
- **Quality**: Sharp edges, preserves fine details
- **Best For**: Pixel art, geometric images, when preserving sharp details is crucial
- **Note**: May lose some information but maintains crisp boundaries

## Usage

### Basic Command

```bash
mflux-generate-redux-advanced \\
    --prompt "your prompt here" \\
    --redux-image-paths path/to/reference/image.jpg \\
    --redux-mode medium \\
    --interpolation-mode area \\
    --steps 12 \\
    --guidance 4.5 \\
    --output output_image.png
```

### Mode Comparison Example

```bash
# Test all five ComfyUI modes with the same prompt and image
for mode in lowest low medium high highest; do
  mflux-generate-redux-advanced \\
    --prompt "anime style, Studio Ghibli" \\
    --redux-image-paths photo.jpg \\
    --redux-mode $mode \\
    --seed 42 \\
    --output test_${mode}.png
done
```

### New Commands

#### `mflux-generate-redux-advanced`

Enhanced version of the redux command with mode control:

```bash
mflux-generate-redux-advanced \\
    --model black-forest-labs/FLUX.1-Krea-dev \\
    --prompt "your prompt here" \\
    --redux-image-paths path/to/reference/image.jpg \\
    --redux-mode medium \\
    --steps 12 \\
    --guidance 4.5 \\
    --width 896 \\
    --height 1152 \\
    --output output_image.png
```

**New Parameters:**
- `--redux-mode`: Choose between `lowest`, `low`, `medium`, `high`, or `highest` (default: `medium`) - ComfyUI compatible
- `--interpolation-mode`: Choose interpolation method `area`, `bicubic`, or `nearest` (default: `area`, optional) - ComfyUI compatible

### Updated Scripts

#### Enhanced `minimal_memory_efficient.sh`

The main script now supports the advanced Redux mode:

```bash
./minimal_memory_efficient.sh "prompt in italiano" [image_path] [num_images] [redux_mode]
```

**New Parameter:**
- `redux_mode`: lowest|low|medium|high|highest (default: medium)

**Examples:**
```bash
# Minimal influence
./minimal_memory_efficient.sh "una donna bellissima" image.jpg 3 lowest

# Low influence
./minimal_memory_efficient.sh "una donna bellissima" image.jpg 3 low

# Balanced influence (default)
./minimal_memory_efficient.sh "una donna bellissima" image.jpg 3 medium

# High influence
./minimal_memory_efficient.sh "una donna bellissima" image.jpg 3 high

# Maximum influence
./minimal_memory_efficient.sh "una donna bellissima" image.jpg 3 highest
```

#### Test Script

Use the included test script to experiment with different modes:

```bash
./test_redux_advanced.sh [lowest|low|medium|high|highest] [prompt] [image_path] [num_images]
```

### Enhanced Metadata

Generated images now include the redux mode in their metadata:

```json
{
  "redux_mode": "medium",
  "redux_image_paths": "['/path/to/reference/image.jpg']",
  "redux_image_strengths": [1.0]
}
```

## Technical Implementation

### Core Classes

1. **`ReduxUtilAdvanced`**: Enhanced image embedding with mode-specific transformations
2. **`Flux1ReduxAdvanced`**: Advanced Redux model with configurable influence control
3. **`GeneratedImage`**: Updated to include redux mode metadata

### Key Files Modified

- `src/mflux/flux_tools/redux/redux_util_advanced.py` - New advanced Redux utilities
- `src/mflux/flux_tools/redux/redux_advanced.py` - New advanced Redux model
- `src/mflux/generate_redux_advanced.py` - New command implementation
- `src/mflux/post_processing/generated_image.py` - Added redux_mode support
- `src/mflux/post_processing/image_util.py` - Added redux_mode parameter
- `pyproject.toml` - Added new command entry point

## Usage Recommendations

### For Portraits and Character Art
- **Lowest**: Minimal subject influence, maximum creative interpretation
- **Low**: Light subject guidance with creative freedom
- **Medium**: Balanced results that keep key features while allowing style variation
- **High**: Strong subject adherence with some creative flexibility
- **Highest**: Maximum fidelity to subject features and details

### For Style Transfer
- **Lowest**: Subtle style hints while preserving prompt direction
- **Low**: Light style influence with creative interpretation
- **Medium**: Balanced style adoption with prompt flexibility
- **High**: Strong style adherence with some prompt influence
- **Highest**: Maximum style fidelity, almost like direct style transfer

### For Composition Reference
- **Lowest**: Minimal compositional hints
- **Low**: Light compositional guidance
- **Medium**: Balanced composition with creative freedom
- **High**: Strong compositional adherence
- **Highest**: Strict adherence to reference composition and layout

## Examples

### Lowest Mode Example
```bash
mflux-generate-redux-advanced \\
    --prompt "a futuristic cityscape at sunset" \\
    --redux-image-paths classical_architecture.jpg \\
    --redux-mode lowest
```
Result: A futuristic cityscape with minimal classical architectural hints

### Highest Mode Example
```bash
mflux-generate-redux-advanced \\
    --prompt "a modern interpretation" \\
    --redux-image-paths mona_lisa.jpg \\
    --redux-mode highest
```
Result: A modern style image that closely follows the Mona Lisa's composition and pose

## Performance Notes

- All modes have similar computational requirements
- Lowest mode provides maximum prompt flexibility
- Highest mode provides maximum image fidelity
- No significant performance difference between modes

## Compatibility

- Fully backward compatible with existing Redux functionality
- Can be used with all existing LoRA configurations
- Supports all standard FLUX model variants
- Works with existing callback and memory management systems
