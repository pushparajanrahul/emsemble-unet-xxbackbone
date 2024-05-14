# Medical Image Segmentation Ensemble

This repository contains an ensemble technique for medical image segmentation using semantic segmentation models with different backbones. The ensemble approach combines predictions from multiple models to improve segmentation accuracy.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Installation](#installation)
- [Requirements](#requirements)
- [Data](#data)
- [Models](#models)
- [Examples](#examples)
- [License](#license)

## Overview

Medical image segmentation plays a crucial role in various healthcare applications, including disease diagnosis and treatment planning. This project aims to enhance the accuracy of medical image segmentation by combining predictions from multiple semantic segmentation models with different backbones. The ensemble technique aggregates the outputs of individual models, leveraging their diverse strengths to achieve better segmentation results.

## Usage

To use the ensemble technique for medical image segmentation:

1. Clone this repository to your local machine.
2. Install the required dependencies (see [Installation](#installation)).
3. Prepare your medical image dataset (see [Data](#data)).
4. Train or load pre-trained semantic segmentation models (see [Models](#models)).
5. Modify the `ensemble_segmentation.py` script to specify the models and their weights in the ensemble.
6. Run the `ensemble_segmentation.py` script to perform segmentation on ChestXDet dataset.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Requirements

Ensure you have the following dependencies installed:

- TensorFlow
- Keras
- segmentation_models
- scikit-learn
- OpenCV
- matplotlib
- NumPy
- pandas

## Code Structure

The repository has the following structure:

```
CSE598
├── ChestXDet
│   ├── ChestXDet_Metainformations
│   │   └── ChestX-Det-Dataset-main
│   │       └── pre-trained_PSPNet
│   │           └── ptsemseg
│   ├── test_data
│   │   ├── mask
│   │   └── test
│   └── train_data
│       ├── mask
│       └── train
├── savemodels
└── src
```

- **ChestXDet/**: Directory for ChestXDet dataset. Contains subdirectories for training and testing data.
- **savemodels/**: Directory to save trained models.
- **src/**: Contains the source code for the ensemble segmentation technique along with the utility file.
  - `ensemble_segmentation.py`: Python script implementing the ensemble technique.
  - `utils.py`: Utility functions for creating the coco-dataset and mask generation (we are generating the masks from the metadata information (.json) provided along the Dataset).

  
## Data

You can use your own medical image dataset or obtain one from public repositories. Organize your dataset in the following structure:
```
ChestXDet
├── ChestXDet_Metainformations
│   └── ChestX-Det-Dataset-main
│       └── pre-trained_PSPNet
│           └── ptsemseg
├── test_data
│   ├── mask
│   └── test
└── train_data
    ├── mask
    └── train
```

- **ChestXDet_Metainformations**: Contains additional information about the ChestXDet dataset such as test and train data annotation in .json format.
- **test_data**: Directory for testing images and corresponding masks.
- **train_data**: Directory for training images and corresponding masks.

Ensure that the images and corresponding masks are properly aligned.

## Usage

To use the ensemble segmentation technique:
1. Clone the repository: `git clone <repository-url>`
2. Create a virtual enviroment(recommended) and install dependencies: `pip install -r requirements.txt`
3. Download the ChestXDet dataset from here to the downloaded repository folder and ake sure so that the images and corresponding masks folders are properly aligned.
4. Navigate to the `src` directory: `cd src`
5. Run the `utli.py` to generate the masks from the metadata informations available from .json files. Verify the generated masks by navigating to the masks folder.
6. Run the ensemble segmentation script: `python ensemble_segmentation.py`.

## Results

The results observed for the given hyperparameters are discussed below:

```
EPOCH=100
SIZE_X = 128 
SIZE_Y = 128
batch_size=8
```


## Evaluation Metrics
We evaluated the performance of individual models and the ensemble technique using the Intersection over Union (IOU) score on a test dataset. The IOU score measures the overlap between predicted and ground truth masks, providing insight into segmentation accuracy.

### Individual Model Performance
- **Model 1 (ResNet34 Backbone)**:
  - IOU Score: 0.1215
- **Model 2 (EfficientNetB4 Backbone)**:
  - IOU Score: 0.0002
- **Model 3 (VGG16 Backbone)**:
  - IOU Score: 0.1619

### Ensemble Technique Performance
- **Weighted Average Ensemble**:
  - IOU Score: 0.1189

## Weight Optimization
We performed grid search to find the optimal combination of weights for the ensemble technique. The maximum IOU score of 0.1629 was obtained with the following weights:
- Model 1: 0.6
- Model 2: 0.5
- Model 3: 0.8

## Conclusion
- Model 3 (VGG16 Backbone) achieved the highest IOU score among individual models.
- The ensemble technique did not significantly outperform individual models, possibly due to imbalanced model contributions.
- Further experimentation with weight optimization and model architectures may improve segmentation accuracy.

## Prediction
![image](https://github.com/pushparajanrahul/ensemble-segmentation/assets/124497777/8222bf84-ce9b-4f89-8056-76185b899ece)


## License

This project is licensed under the [MIT License](LICENSE).
