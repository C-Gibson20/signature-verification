# Triplet Loss Signature Verification

This project implements an offline signature verification system using deep metric learning. It trains a **ResNet50-based Siamese triplet network** with **semi-hard triplet loss**, enabling robust distinction between genuine and forged signatures. Training is accelerated via **TPU**, and generalization is improved through **data augmentation** and **K-Fold cross-validation**.

<br>

## Overview

### Data Preprocessing

* Input images are read in **grayscale**, resized to **(220, 155)**, and binarized using **Otsu's thresholding**.
* Inverted binary images are used to enhance signature foreground.
* Writer IDs ≤ threshold (default: 11) are used for test split; the rest for training.

### Data Pipeline

* **Triplet Generator** yields batches of (anchor, positive, negative) images.
* Triplets are sampled per writer:

  * Anchor & Positive: different genuine samples of the same writer.
  * Negative: a forgery or sample from the same writer’s forgery set.
* Images are broadcast to 3 channels and normalized.

### Augmentation

Augmentations improve model invariance to handwriting variation:

* **Affine transforms:** rotation (±5°), scaling (±10%), and translation (±10%).
* **Elastic deformation:** mimics pen pressure and stroke variations.

<br>

## Model Architecture

### Embedding Network

* **Backbone:** `ResNet50` (`include_top=False`, `imagenet` weights).
* Last 30 layers unfrozen for fine-tuning.
* **Intermediate output:** `conv4_block6_add`.
* Followed by:

  * Two Conv-BN-Dropout blocks (with L2 regularization).
  * Skip connection via `Conv2D(1x1)` + BatchNorm.
  * GlobalAveragePooling → Dense(256) → `L2-normalized` embeddings.

> Output: `256-D` unit-normalized embeddings suitable for cosine similarity.

### Loss Function: Semi-Hard Triplet Loss

A custom implementation that:

* Computes pairwise cosine distances between embeddings.
* Forms triplets satisfying the semi-hard condition:

  * Positive closer than negative but within margin.
* Applies the margin-based ranking loss only on valid triplets.

### Custom Training Loop (`tf.keras.Model` subclass)

* `train_step`: embeds anchor, positive, negative → compute loss → backprop.
* `test_step`: evaluates similarity and computes binary accuracy.
* `custom_predict()`: computes cosine similarity between arbitrary image pairs.

<br>

## Training Strategy

### TPU Training

* TPU initialized via `TPUClusterResolver`.
* All model creation, compilation, and training wrapped in `tpu_strategy.scope()`.

### Optimization

* Optimizer: `Adam`, learning rate `0.001` (decayed via `ReduceLROnPlateau`).
* Metrics:

  * `train_loss`: averaged triplet loss.
  * `val_accuracy`: binary classification based on cosine similarity threshold.

### K-Fold Cross-Validation

* `K=4` folds with shuffled writer IDs.
* For each fold:

  * Checkpointing via `ModelCheckpoint`.
  * Early stopping and TensorBoard logging.
  * Best fold selected by max validation accuracy.

<br>

## Evaluation

* Best model loaded from checkpoint of top-performing fold.

* Evaluated on held-out test set.

* Achieved final **validation accuracy**:

  ```
  83.89%
  ```

* Inference is performed by computing cosine similarity between anchor and test signature embeddings and thresholding the result.

