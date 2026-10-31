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

## 4. Model Architectures

### 4.1 CNN Baseline

A compact CNN is implemented as the baseline for comparison with the GFNet-inspired model. It receives a single-channel 256 × 256 MRI slice and produces two logits corresponding to NC and AD.

The feature extractor contains three convolutional stages. Each stage consists of a 3 × 3 convolution with stride 1 and padding 1, a ReLU activation, and a 2 × 2 max-pooling operation with stride 2. The channel dimensions increase from 1 to 32, 64, and 128.

| Component | Configuration | Output shape, excluding batch |
|---|---|---|
| Input | Grayscale MRI slice | 1 × 256 × 256 |
| Convolutional stage 1 | Conv 1 → 32, ReLU, max pooling | 32 × 128 × 128 |
| Convolutional stage 2 | Conv 32 → 64, ReLU, max pooling | 64 × 64 × 64 |
| Convolutional stage 3 | Conv 64 → 128, ReLU, max pooling | 128 × 32 × 32 |
| Global average pooling | Adaptive average pooling to 1 × 1 | 128 × 1 × 1 |
| Classifier | Flatten and linear projection | 2 |

Global average pooling summarises each feature channel over the spatial dimensions. A linear layer maps the resulting 128-dimensional representation to the two output logits. This avoids a large fully connected layer operating on flattened spatial feature maps.

The baseline contains **92,930 trainable parameters**. Its compact architecture provides a reference for assessing whether the additional complexity of the GFNet-inspired model improves classification performance.

### 4.2 GFNet-Inspired Architecture

The advanced model uses learnable frequency-domain filters to mix information across a spatial grid of patch features. The implementation follows the global filtering concept of GFNet and uses separate residual connections around the filtering and MLP branches.

Each 256 × 256 input image is divided into non-overlapping 16 × 16 patches using a convolution with kernel size 16 and stride 16. Each patch is projected into a 128-dimensional feature vector, producing a 16 × 16 feature grid containing 256 patch representations.

A learnable positional embedding is added to this grid, followed by dropout. The features then pass through eight global filter blocks. A final layer normalization and spatial average pooling produce a 128-dimensional image representation, which is mapped to two output logits.

| Component | Configuration |
|---|---|
| Input | Single-channel 256 × 256 MRI slice |
| Patch embedding | Conv2d, kernel size 16, stride 16, 128 output channels |
| Feature grid | 16 × 16 spatial positions |
| Positional embedding | Learnable tensor of shape 1 × 16 × 16 × 128 |
| Global filter blocks | 8 |
| Embedding dimension | 128 |
| MLP hidden dimension | 512 |
| Activation | GELU |
| Dropout probability | 0.1 |
| Final normalization | LayerNorm |
| Pooling | Average over the spatial grid |
| Classifier | Linear projection from 128 features to 2 logits |

The model contains **1,418,882 trainable parameters**, approximately 15.3 times the parameter count of the CNN baseline.

### 4.3 Global Filter Layer

The global filter operates independently on each feature channel. First, a two-dimensional real FFT transforms the spatial feature map into the frequency domain. The resulting coefficients are multiplied element-wise by learnable complex-valued weights. An inverse FFT then converts the filtered features back into the spatial domain:

$$
G(X) =
\mathcal{F}^{-1}
\left(
W \odot \mathcal{F}(X)
\right),
$$

where $\mathcal{F}$ denotes the two-dimensional Fourier transform, $W$ contains the learnable complex-valued filter weights, and $\odot$ denotes element-wise multiplication.

In the implementation, `torch.fft.rfft2` and `torch.fft.irfft2` use orthonormal normalization. For a 16 × 16 spatial grid, the stored real-FFT representation has dimensions 16 × 9 per channel. The real and imaginary components of the filter weights are stored in a parameter tensor of shape 128 × 16 × 9 × 2.

This operation provides global spatial mixing within each channel. The subsequent MLP mixes information across feature channels at each spatial position.

### 4.4 Residual Block Design

Each block contains two residual connections. The first surrounds the global filter branch, while the second surrounds the MLP branch:

$$
Y = X + G\left(\operatorname{LN}_1(X)\right),
$$

$$
Z = Y + \operatorname{MLP}\left(\operatorname{LN}_2(Y)\right).
$$

Here, $\operatorname{LN}_1$ and $\operatorname{LN}_2$ are separate layer normalization operations. The MLP expands the feature dimension from 128 to 512, applies GELU and dropout, and projects the features back to 128 dimensions before a second dropout operation.

The residual connections provide identity paths alongside the learned transformations. Their individual contribution to performance has not been established through a controlled ablation experiment.

### 4.5 Classification Outputs

Both models return two unnormalized logits. Class 0 represents NC, and class 1 represents AD. Cross-entropy loss is applied directly to the logits during training.

For evaluation, the predicted class is obtained using `argmax`. Softmax is applied to obtain class probabilities, and the probability assigned to AD is used to calculate AUROC. These probabilities are model confidence scores and are not assumed to be calibrated estimates of diagnostic reliability.

## 5. Experimental Setup

### 5.1 Shared Experimental Conditions

Both models use the same patient-independent training, validation, and test partitions described in Section 3. They also share the same input resolution, normalization, training augmentation pipeline, and batch size.

Experiments are conducted in Google Colab using an NVIDIA A100-SXM4 GPU with 40 GB of memory. Images are loaded from local Colab storage after extraction from Google Drive.

| Setting | CNN baseline | GFNet-inspired model |
|---|---|---|
| Input size | 1 × 256 × 256 | 1 × 256 × 256 |
| Batch size | 32 | 32 |
| Configured training epochs | 30 | 30 |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | Adam | AdamW |
| Initial learning rate | 0.001 | 0.0003 |
| Weight decay | 0 | 0.0005 |
| Learning-rate scheduler | None | CosineAnnealingLR |
| Minimum learning rate | Not applicable | 0.000001 |
| Checkpoint selection | Lowest validation loss | Lowest validation loss |

Training data are shuffled at each epoch. Validation and test data are evaluated without shuffling or random augmentation. Each DataLoader uses two worker processes and pinned memory.

The same split and evaluation procedures support a comparison of the two implemented systems. However, the optimizers and learning-rate schedules differ, so performance differences cannot be attributed exclusively to model architecture.

### 5.2 Training Procedure

Both models are trained from scratch. For each training batch, the model produces two logits, and cross-entropy loss is calculated against the integer class labels. Gradients are computed through backpropagation, and the optimizer updates the model parameters.

Epoch loss is calculated as the sample-weighted average of batch losses:

$$
L_{\mathrm{epoch}}
=
\frac{\sum_{b=1}^{B} n_b L_b}
{\sum_{b=1}^{B} n_b},
$$

where $L_b$ is the mean loss for batch $b$, and $n_b$ is the number of samples in that batch. This accounts for the potentially smaller final batch.

Accuracy is calculated as the number of correct predictions divided by the total number of evaluated images. Training and validation losses and accuracies are recorded after each epoch.

For GFNet, the learning rate is updated after each epoch using cosine annealing, with `T_max = 30` and `eta_min = 1e-6`. The CNN baseline uses a constant learning rate.

### 5.3 Validation and Checkpoint Selection

After each training epoch, the model is evaluated on the validation set using evaluation mode and disabled gradient computation. Whenever validation loss improves, the model weights are saved.

The best checkpoint is therefore defined by the lowest validation loss, rather than the highest training accuracy or the final training epoch. Before final evaluation, the saved checkpoint is reloaded and the model is placed in evaluation mode.

Although early stopping was considered in the initial feasibility review, the implemented training procedure uses a fixed budget of 30 epochs and retains the checkpoint with the lowest validation loss.

### 5.4 Reproducibility

A random seed of 42 is used for patient-level splitting and for NumPy and PyTorch random number generation. CUDA random seeds are also set. The CNN notebook additionally seeds Python's `random` module.

Seeds help reduce variation between runs, but the experiments do not enforce fully deterministic GPU execution. Consequently, exact numerical reproducibility across hardware or software environments is not guaranteed.

### 5.5 Evaluation Protocol

The selected checkpoints are evaluated on the same test set of 4,440 MRI slices from 102 patients. Metrics are calculated at the slice level.

The evaluation includes:

- Overall accuracy.
- Per-class precision, recall, and F1-score.
- Macro-averaged F1-score.
- Balanced accuracy.
- AUROC using the predicted probability of AD.
- Confusion matrices and ROC curves.

AD is treated as the positive class. Therefore, AD recall measures the proportion of AD-labelled slices correctly classified, while NC recall measures the proportion of NC-labelled slices correctly classified.

The final partitions are patient-independent. However, test results were inspected during exploratory development and informed some subsequent experimental choices. The reported test performance therefore has a limitation associated with repeated test-set feedback and should not be interpreted as a completely untouched confirmatory evaluation.