
 XPACT: PCB X-ray CT Image Dataset
==================================

Dataset Structure
-----------------
This dataset contains high-resolution X-ray CT scan data of multiple classes of Printed Circuit Boards (PCBs), captured using industrial CT equipment. The dataset is organized into three major categories based on imaging modality:

1. `2D_CT_slices_XY_plane/`  
   - Contains 2D axial CT slices (in `.zip` format) for each PCB class and instance.
   - Subfolders group PCBs by board type, such as `Arduino_duplicate/`, `Basys3_FPGA/`, `RaspberryPi4/`, etc.

2. `dicom_images/`  
   - Contains 3D DICOM volumes of the same PCBs.
   - Useful for volumetric reconstruction and deep 3D learning tasks.

3. `projection_images/`  
   - Contains raw projection images (similar to X-ray views at multiple angles).
   - These are valuable for testing tomographic reconstruction or object detection techniques.

Each zipped file contains either `.tif`, or `.dcm` images depending on the modality.

File Format Summary
-------------------
- 2D CT slices: `.tif` (zipped)
- DICOM volumes: `.dcm` (zipped)
- Projection images: `.tif` (zipped)

Dataset Importance
------------------
This dataset is designed for research in:
- PCB reverse engineering
- Anomaly detection and tamper detection in hardware security
- Deep learning models for tomography and segmentation
- Cross-modal learning (2D + 3D)

The dataset includes various manufacturers, orientations, and board types — allowing for broad benchmarking and generalization studies.

Accessing the Dataset in Code
-----------------------------

To use the images from Hugging Face in this GitHub-hosted deep learning code, follow these steps:

### Step 1: Install `huggingface_hub`

```bash
pip install huggingface_hub

### Step 2: Log in to Hugging Face

```bash
huggingface-cli login     (Use our token with read access.)

### Step 3: Clone the dataset locally

```bash
git lfs install
git clone https://huggingface.co/datasets/SEAL-IIT-KGP/XPACT

Or use the Hub directly in Python:

```python
from huggingface_hub import hf_hub_download

# Example: download a specific file
file_path = hf_hub_download(
    repo_id="SEAL-IIT-KGP/XPACT",
    filename="2D_CT_slices_XY_plane/Arduino_original/Arduino1_original_set1.zip"
)

 Additionally, this repository includes the codebase of our recently published work "X-Factor: Deep Learning-based PCB Counterfeit Detection using X-ray CT Techniques for Hardware Assurance"

 The codebase corresponds to five different Convolutional Neural Network (CNN) models which are transfer-learned on ImageNet dataset and fine tuned for the specific X-ray CT dataset. 


 ## Citation

If you use this code or data in your research, please cite:

➤ [BibTeX](xfactor.bib)

