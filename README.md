# F2NetViT GitHub Repository

This repository contains supplementary files associated with the research project titled "Evaluating the Performance of nnU-Net Architectures on BraTS 2024 Challenge Datasets." The files include visualisation code outputs, model predictions, and other resources that support the findings and methodologies presented in the study.

## Repository Overview

This repository is structured into several key directories, each containing specific files related to the project:

### 1. Visualisation Code Outputs

These directories contain visualisation outputs used to analyse and compare the segmentation results of different models:

- **Pred vs. GT**: This directory includes visualisations comparing model predictions with ground truth segmentations, useful for assessing model accuracy and precision across various tumour regions.

- **Modality Comparison With GT**: Contains visualisations that compare model predictions across four different MRI modalities. This helps evaluate how effectively the model utilises multi-modality information for accurate segmentation.

- **Multiclass Comparison**: Provides visualisations for comparing segmentation outputs across different tumour classes, highlighting the model's ability to differentiate between various types of tumour tissue.

- **Models Compared**: Showcases visualisations where models performed well or poorly in segmenting tumour regions. These examples help understand the strengths and weaknesses of each model in handling complex segmentation tasks.

- **Intel 2D Model Predictions**: Contains visualisations specific to the Intel 2D U-Net model's predictions, useful for evaluating the model's performance in simpler segmentation tasks.

- **8x8 Model Visualisation**: Includes an 8x8 grid of visualisations showing segmentation results across multiple MRI slices, providing a comprehensive view of model performance across different brain sections.

### 2. Model Code

These directories contain the source code for the different models used in the study:

- **nnU-Net**: Contains the code for the 2D and 3D nnU-Net models designed for multiclass segmentation tasks, suitable for handling complex segmentation scenarios with high accuracy.

- **2D Intel**: Includes the code for the Intel 2D U-Net model, optimised for binary segmentation tasks and suitable for environments with limited computational resources.

## Instructions for Use

1. **Clone the Repository**: To get started, clone the repository to your local machine using the following command:
    ```bash
    git clone https://github.com/FahimR12/F2NetViT.git
    ```

2. **Navigate to the Desired Directory**: Once cloned, navigate to the directory of interest to access the files. Each directory includes a specific README file with detailed instructions on how to use the scripts and interpret the visualisations.

3. **Running the Visualisation Scripts**: 
    - Ensure you have all the required dependencies installed as specified in the `requirements.txt` file.
    - Follow the instructions provided in the README files within each visualisation directory to run the scripts and generate visual outputs.

4. **Using the Model Code**: 
    - The `nnU-Net` and `2D Intel` directories contain subfolders with model code and training scripts. Refer to the README files within these directories for details on how to set up the environment, train the models, and evaluate their performance.

## Additional Information

For further details on the methodologies and results, please refer to the main project documentation and appendix sections in the research report. If you encounter any issues or have questions, feel free to open an issue in the repository.

Happy coding!
