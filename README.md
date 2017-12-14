# automatic_screening_nma
Code and dataset for the project: Automatic screening for updating Network meta-analysis

Most of the code is from https://github.com/dennybritz/cnn-text-classification-tf.git

Minor changes :
* different evaluation metrics: sensitivity and specificity
* different batch function : balanced batch (rare class oversampling), with sampling with replacement
* text is already vectorized as followed : each citations is represented as the average of its word embeddings
