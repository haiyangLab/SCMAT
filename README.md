# SCMAT
SCMAT, a deep learning method based on the MLP and Transformer Block for extracting low-dimensional feature representations of complex multi-omics data. SCMAT can 
integrate transcriptome and proteome data from the same cell and transcriptome and epigenome data, describing single-cell multi-omics data with different distributions
by integrating multiple-input multiple-output networks. The Transformer model is a popular advanced network model. To the best of our knowledge, SCMAT is the first model 
to apply the Transformer to clustering analysis of single-cell multi-omics data. Additionally, Gaussian mixture model clustering was used to obtain accurate cell cluster 
types. We applied SCMAT to four real and three simulated datasets, achieving optimal performance compared to other advanced clustering methods. We found that integrating 
multi-omics data improved the performance of our method compared to using only individual omics data. Lastly, the high operational efficiency and low overhead of SCMAT 
make it possible to extend the method to analyze large multi-omics datasets, thus making it a promising tool for clustering and analyzing vast amounts of data. Through 
detailed experimental analysis, we demonstrate the effectiveness of SCMAT.

## We can run the SCMAT model with the following command(Example):
```bash
python train.py -i ./input/simulate3.list -m SCMAT -t simulate3 -model transformer -e 300
```
Simulate3 is a simulated dataset and it has 3 cell types.

## Evaluation
The performance analysis of the results was conducted using R language, and the following are the analysis steps:
1. Construct cell labels
2. Organize the output of the model
3. Using `Evaluation. R` to complete performance evaluation

## Settings
SCMAT's model implementation is based on Pytorch. Its dependency packages are: Python (3.7.10), PyTorch (1.7.1), NumPy (1.20.3), Pandas (1.3.4), Keras (2.3.1), Scipy(1.7.1). The operating system is windows10. The GPU is NVIDIA GeForce GTX 3090.
