# ISIC Challenge Training Code

This repository contains code for training a model for the ISIC Challenge. The ISIC Challenge is a competition focused on skin lesion classification, organized by the International Skin Imaging Collaboration.

## ISIC 2019 Challenge Information

The ISIC 2019 challenge is a competition focused on skin lesion classification, organized by the International Skin Imaging Collaboration. It aims to advance the field of dermatology and improve the diagnosis of skin lesions through machine learning and computer vision techniques.

### Overview

- **Organizer:** International Skin Imaging Collaboration (ISIC)
- **Year:** 2019
- **Objective:** Skin Lesion Classification
- **Dataset:** The challenge provides a dataset of skin lesion images for training and evaluation.
- **Task:** Participants are tasked with developing machine learning models to classify skin lesions into different categories, such as melanoma, nevus, and seborrheic keratosis.

### Dataset

The ISIC 2019 challenge dataset consists of high-resolution images of skin lesions captured using various imaging modalities. Ground truth labels provided by dermatologists for each image indicate the diagnosis or classification of the skin lesion.

### Evaluation Criteria

Participants' models are evaluated based on accuracy, balanced accuracy, and other metrics such as sensitivity, specificity, and area under the receiver operating characteristic curve (AUC-ROC).

### Results

The results of the ISIC 2019 challenge showcase the performance of different machine learning models and techniques in skin lesion classification. Participants' submissions are ranked based on their performance on the evaluation metrics.

### Conclusion

The ISIC 2019 challenge provides a platform for researchers and practitioners to develop and evaluate state-of-the-art machine learning algorithms for skin lesion classification. By advancing the field of dermatology, these efforts contribute to the early detection and diagnosis of skin cancer and other dermatological conditions.

## Code Design Date

The design of this training code was completed on January 17, 2020.

## Requirements

To run the code, you need to have the following dependencies installed:

- Python 3.x
- PyTorch
- Albumentations
- OpenCV
- Apex (for mixed precision training)
- scikit-learn
- pandas

## Usage

You can run the training script (`main.py`) from the command line. Here are some usage examples:

### Training

To train the model:

```bash
python main.py /path/to/dataset --folds N --val-fold N --seed SEED --epochs EPOCHS --batch-size BATCH_SIZE --lr LEARNING_RATE --arch ARCHITECTURE --model-type MODEL_TYPE --opt-method OPTIMIZER_METHOD
```
- `/path/to/dataset`: Path to the dataset directory.
- `--folds N`: Number of cross-validation folds.
- `--val-fold N`: Validation fold.
- `--seed SEED`: Cross-validation random seed.
- `--epochs EPOCHS`: Number of total epochs to run.
- `--batch-size BATCH_SIZE`: Mini-batch size.
- `--lr LEARNING_RATE`: Initial learning rate.
- `--arch ARCHITECTURE`: Path to the pre-trained model checkpoint or the pre-trained model name.
- `--model-type MODEL_TYPE`: Model type (e.g., efficientnet-b0).
- `--opt-method OPTIMIZER_METHOD`: Optimizer method (SGD, Adam, RAdam).

### Evaluation

To evaluate the trained model:

```bash
python main.py /path/to/dataset --evaluate --arch ARCHITECTURE

```

- `--evaluate`: Evaluate model on the validation set.

## Contributing

Contributions to improve this codebase are welcome! Please feel free to submit issues or pull requests.

## License

This code is released under the MIT License. 

