# ADNI MRI Classification Using a GFNet-Inspired Model and a CNN Baseline

## 1. Project Overview

This project investigates the classification of two-dimensional brain MRI slices into Alzheimer’s disease (AD) and normal control (NC) categories using the ADNI dataset. A GFNet-inspired model is implemented and compared with a simple convolutional neural network (CNN) baseline. Both models use the same patient-independent training, validation, and test splits, ensuring that all images belonging to a patient remain within a single partition. Predictions and performance metrics are evaluated at the slice level.

The GFNet-inspired model divides each image into patches and uses learnable frequency-domain filters to mix information across the spatial feature grid. Each block contains separate residual connections around the global filtering operation and the multilayer perceptron. The CNN baseline uses three convolutional stages followed by global average pooling and a classification layer. This comparison investigates whether global frequency-domain processing provides useful improvements over a compact convolutional model.

The evaluation addresses both classification performance and the confidence and ambiguity dilemma in automated triage. Accuracy, per-class precision and recall, macro-F1, and AUROC are used to compare predictive performance. Further analysis examines representative misclassifications, prediction confidence, and computational cost to assess the trade-off between model complexity and practical utility. The intended outcome is an evidence-based recommendation on the prototype’s limitations and the conditions under which predictions should be referred for human review.

## 2. Feasibility Review — GFNet for Alzheimer's Disease Classification
This section records the initial feasibility review prepared for the midpoint check-off. It describes the proposed approach and preliminary evidence at that stage; the final implementation and experimental results are documented in the subsequent sections. CN is referred to as NC elsewhere in this report, following the dataset folder labels.

**COMP3710 Open-Ended Project | Domain: Medical Image Classification | Difficulty: Hard**

### 2.1 User Need, Scope & Acceptance Criteria

This project aims to develop a deep learning prototype to assist medical reviewers in distinguishing Alzheimer's Disease (AD) from Cognitively Normal (CN) cases using 2D T1-weighted brain MRI slices from the ADNI dataset. The system will compare a conventional CNN baseline with a Global Filter Network (GFNet), while investigating prediction confidence and identifying uncertain cases requiring expert review.

**Acceptance criteria:**

- Achieve at least 80% test accuracy for AD versus CN classification.
- Compare GFNet against a CNN baseline using accuracy, per-class precision, recall, macro-F1, and AUROC.
- Prevent patient-level data leakage across training, validation, and test sets.
- Evaluate prediction confidence and investigate 3–5 representative failure cases.
- Establish an empirical confidence threshold for referring uncertain predictions to human reviewers, and report inference latency and GPU memory usage.

### 2.2 Model Choice & Course Concepts

The project uses the ADNI dataset, comprising 2D MRI slices labelled AD or CN. A simple CNN will serve as the baseline, while GFNet is the selected advanced architecture (Hard Difficulty).

GFNet replaces conventional self-attention with global frequency-domain filtering using the 2D Fast Fourier Transform (FFT), learnable frequency-domain filters, and inverse FFT. This approach enables global spatial information mixing. Relevant course concepts include convolutional neural networks, Fourier transforms, backpropagation, classification metrics, and model generalisation.

Both models will be trained and evaluated using the same patient-independent data partitions.

### 2.3 Preliminary Feasibility Evidence

Initial data inspection identified 10,400 AD and 11,121 CN training slices, together with 4,460 AD and 4,540 CN test slices. Images are grayscale with a resolution of 256 × 240 pixels.

Metadata analysis identified 680 unique subjects, with 216 subjects appearing in both the supplied training and test partitions. All examined image identifiers were successfully mapped to metadata, and no conflicting AD/CN subject labels were detected.

This demonstrates a significant patient-level leakage risk in the supplied split. A new stratified, subject-level 70/15/15 training, validation, and test partition is proposed. Model training and computational feasibility testing remain pending.

### 2.4 Risks, Budget & Fallback

**Key risks:** Patient-level leakage, limited GPU availability, GFNet training cost, overfitting, and unreliable confidence estimates.

**Mitigation:** Use patient-independent partitions, reduced image resolution where appropriate, smaller training batches, early stopping, and reproducible experiments.

**Computing budget:** Google Colab GPU resources, with peak VRAM, epoch duration, and inference latency to be measured during initial experiments.

**Next experiment:** Construct the patient-independent data splits, train a small CNN baseline, and benchmark GFNet training feasibility on a limited subset.

**Fallback:** If the full GFNet architecture exceeds available resources, use a reduced-width GFNet retaining its core Fourier filtering mechanism, and report the computational trade-offs.

## 3. Dataset and Preprocessing

### 3.1 Dataset Overview and Data Audit

This project uses the preprocessed ADNI dataset provided for the course, containing two-dimensional grayscale MRI slices labelled as Alzheimer’s disease (AD) or normal control (NC). The initial inventory contained 30,521 image records from 680 unique patients. Patient identifiers were extracted from the metadata and associated with individual image records using their image IDs. All image records were successfully mapped to patients, and no conflicting class labels were found within patients.

An audit of the original training and test folders identified 216 patients appearing in both partitions. Therefore, the original folder assignments were not used as the final experimental splits. One previously identified duplicate record, `976382_96 (1).jpeg`, was removed, leaving 30,520 images for subsequent experiments.

![Example AD and NC MRI slices](dataset/report/adni_examples.png)

*Figure 1. Example grayscale MRI slices from the provided ADNI dataset: AD (left) and NC (right), shown before resizing and training augmentation. Labels follow the dataset annotations.*

### 3.2 Patient-Level Data Splitting

The dataset was repartitioned using class-stratified sampling at the patient level. First, 70% of patients were assigned to training. The remaining 30% were divided equally into validation and test sets. Both splitting operations used a random seed of 42.

All images belonging to the same patient were assigned to one partition. This prevents slices or repeated scans from a patient appearing across training, validation, and test sets. Assertions verified that patient identifiers and image paths did not overlap between partitions, and that all retained patients and image records were included.

| Split | Patients | NC patients | AD patients | NC images | AD images | Total images |
|---|---:|---:|---:|---:|---:|---:|
| Training | 476 | 321 | 155 | 11,000 | 10,620 | 21,620 |
| Validation | 102 | 69 | 33 | 2,220 | 2,240 | 4,460 |
| Test | 102 | 69 | 33 | 2,440 | 2,000 | 4,440 |
| Total | 680 | 459 | 221 | 15,660 | 14,860 | 30,520 |

The 70/15/15 proportions refer to patients rather than images. Image-level proportions differ because patients contribute different numbers of images. Stratification preserves the approximate patient-level class distribution; it does not enforce an equal number of AD and NC images.

Both the CNN baseline and the GFNet-inspired model use these same splits. Validation loss is used to select the best checkpoint within each training run. The selected checkpoint is subsequently evaluated on the test set.

### 3.3 Image Preprocessing

Images are loaded as single-channel grayscale inputs and resized to 256 × 256 pixels. Pixel intensities are converted to floating-point tensors and normalized using a mean of 0.5 and a standard deviation of 0.5:

$$
x_{\mathrm{normalized}} = \frac{x - 0.5}{0.5}
$$

Here, $x$ is the pixel intensity after conversion to the range $[0,1]$. This maps intensities to $[-1,1]$. The same deterministic preprocessing is applied to validation and test images.

For efficient data loading, the image archives are extracted from Google Drive to local Colab storage. Local paths preserve each image’s original relative folder location, independently of its new split assignment. File-existence checks confirm that all images referenced by the final splits are available.

### 3.4 Training Data Augmentation

The same augmentation pipeline is used for both models. Augmentation is applied only to training images.

| Augmentation | Configuration |
|---|---|
| Random affine transformation | Rotation up to ±10°, translation up to 5% in each direction, and scale factor between 0.98 and 1.02 |
| Random erasing | Probability 0.4; area fraction 0.01–0.10; aspect ratio 0.5–2.0 |
| Elastic transformation | Probability 0.3; alpha = 10.0; sigma = 3.0 |

Affine transformation is applied after resizing. Random erasing and elastic transformation are applied after tensor conversion and before normalization. Bilinear interpolation and zero filling are used for affine and elastic transformations.

These augmentations introduce variation in image position, scale, visibility, and local geometry. Their individual contributions have not been isolated through ablation experiments, so this report does not attribute performance improvements to any single augmentation.

