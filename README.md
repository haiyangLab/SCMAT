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
