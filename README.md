# Renaissance

## RenAIssance Project Test

Welcome to the RenAIssance Project Test repository! This project focuses on leveraging AI techniques for historical document analysis and synthetic image generation, specifically targeting 17th-century Spanish book pages. For a detailed overview of the project, please refer to the PDF documents included in this repository.

### Project Structure

This repository contains the following key folders with evaluation results and related artifacts:

#### `test1`: Layout Recognition of Pages
- **Objective**: The `test1` folder contains the results of Evaluation Test 1, which focuses on layout recognition of historical book pages to identify the main text area.
- **Methodology**:
  - We used YOLOv5, a state-of-the-art object detection model, to perform layout recognition.
  - Transfer learning was applied on a custom dataset of historical book pages to fine-tune YOLOv5 for detecting the main text area.
- **Results**: The evaluation results, including model performance metrics (e.g., precision, recall, mAP), are stored in this folder. These results demonstrate the effectiveness of YOLOv5 in accurately identifying the main text area in 17th-century Spanish book pages.

#### `test3`: Synthetic Image Generation
- **Objective**: The `test3` folder contains the results of Evaluation Test 3, which focuses on synthetic image generation of 17th-century Spanish book pages.
- **Methodology**:
  - A two-step process was employed:
    1. **Background Generation**: Generated aged parchment backgrounds using a diffusion model, finetuned to produce realistic textures (e.g., yellowish-grayish-brown tones, ink stains, worn edges).
    2. **Text Overlay**: Overlaid transcribed text from `Buendia transcription.docx` using the "IM Fell English" font, with parameters tuned to match historical layouts (e.g., 24 lines per page, ~50 characters per line, tight line spacing).
  - The generated images were evaluated against an original reference image using metrics such as Mean Squared Error (MSE), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR).
- **Results**:
  - The evaluation metrics are available in the PDF in this folder.
  - Challenges included computational limitations (M2 Pro with MPS was less efficient than CUDA) and a small dataset (40 images), which constrained the diversity of generated backgrounds.
  - The generated images and evaluation results are stored in `test3`, along with Jupyter Notebooks documenting the process and analysis.
 
### Challenges and Future Improvements

- **Computational Resources**: The project faced computational challenges due to the inefficiency of MPS on M2 Pro compared to CUDA. Future work will leverage CUDA-enabled resources to improve the efficiency of diffusion models.
- **Dataset Size**: The synthetic generation process used a limited dataset of 40 images. Expanding the dataset will enhance the diversity and quality of generated backgrounds.
- **Text Overlay**: The text overlay process has scope for improvement, such as enhancing font contrast, reducing noise in aging effects, and fine-tuning alignment.

### Getting Started

To explore the project:
1. Refer to the PDFs for a detailed project overview.
2. Check the `test1` folder for layout recognition results using YOLOv5.
3. Check the `test3` folder for synthetic image generation results, including Jupyter Notebooks with detailed analysis and evaluation metrics.
4. Ensure you have the necessary dependencies installed (e.g., PyTorch, Diffusers, YOLOv5, Pillow) to run the code.

For any questions or contributions, please open an issue or contact the repository owner.
