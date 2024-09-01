# Triplet Loss Signature Verification with TensorFlow on TPU

<p>This repository contains the implementation of a signature verification system using a Siamese Convolutional Neural Network (CNN) optimized for Google Cloud's TPU. The network uses a triplet loss function to learn the embeddings of input images and is trained using K-Fold cross-validation.</p>

<p>The Convolutional Neural Network (CNN) used in this project is based on the ResNet50 architecture, which has been adapted for the Siamese network structure.<br> 
The network uses a semi-hard triplet loss to ensure that genuine signatures are closer to each other in the embedding space than to forged ones.</p>

<p>The training script uses K-Fold cross-validation to train the model on different subsets of the data. The model is trained for 25 epochs using the Adam optimizer and an initial learning rate of 0.001. The training process is logged using TensorBoard.</p>

<p>After training, the model's performance is evaluated using accuracy metrics. The evaluation script computes the accuracy of the model on the test dataset and optionally displays a confusion matrix. The confusion matrix only displays the results from one fold of the evaluation. The validation accuracy result is determined across all folds of the evaluation.</p>

<p>The model achieves an average validation accuracy of approximately 83.9% across the different folds. The confusion matrix generated after evaluation provides insight into the classification performance of the model.</p>
