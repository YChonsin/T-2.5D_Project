# Supplementary details for "2.5D Transformer: An Efficient 3D Seismic Interpolation Method without Full 3D Training"

This is a files of supplementary details for "2.5D Transformer: An Efficient 3D Seismic Interpolation Method without Full 3D Training".

Paper: https://doi.org/10.48550/arXiv.2511.10033

## S1. Dataset preparation details
\
Table S1 provides the dataset preparation details of the Kerry, Parihaka, and Overthrust datasets.
### Table S1. Dataset preparation details of the Kerry, Parihaka, and Overthrust datasets
| Dataset | Kerry & Parihaka | Overthrust |
|---|---|---|
| Initial extraction (**A**) | A 256×650×186 volume randomly selected from each full dataset | A 141×100×801 volume (Sample 30–170, Xline 601–700, Inline 1–801 from the full synthetic volume) |
| Training block | A 256×650×166 volume extracted from **the first 166 inlines of A** | A 141×80×801 volume extracted from **the first 80 xlines of A** |
| Test block | A 224×224×20 volume extracted from **the last 20 inlines of A** | A 141×20×801 volume extracted from **the last 20 xlines of A** |
| **2D dataset for Stage 1, also for T-2D:** | | |
| Patch size | 40×40 | |
| Stride of sliding window | 20×20 | |
| Total patches in the 2D collection (**B1**) | 63744 | |
| Training set randomly sampled from **B1** | 6000 | |
| Validation set randomly sampled from **B1** | 1500 | |
| **3D dataset for Stage 2:** | | |
| Volume size | 40×40×16 | 40×40×16 |
| Stride of sliding window | 16×16×8 | 16×16×8 |
| Total volumes in the 3D collection (**B2**) | 11700 | 2772 |
| Training set randomly sampled from **B2** | 2000 | 2000 |
| Validation set randomly sampled from **B2** | 500 | 500 |
| **3D dataset for full 3D training:** | | |
| Volume size | 40×40×16 | |
| Stride of sliding window | 16×16×8 | |
| Total volumes in the 3D collection (**B3**) | 11700 | |
| Training set randomly sampled from **B3** | 6000 | |
| Validation set randomly sampled from **B3** | 1500 | |

As for the Overthrust dataset, it is a synthetic volume generated from the publicly available 
SEG/EAGE 3-D Overthrust P-wave velocity model (Aminzadeh et al. 1997). The original velocity 
model consists of 187×801×801 grid points with a spatial sampling interval of 25 m. Assuming 
constant density and normal incidence, the reflection coefficients were calculated from the 
velocity contrasts between adjacent depth samples. Subsequently, the reflectivity series along 
each vertical profile was convolved with a Ricker wavelet to generate a band-limited synthetic 
seismic volume. This procedure follows a simplified convolutional representation of post-stack 
seismic data rather than a full wave-equation-based simulation. Therefore, the resulting data 
are intended to represent reflection-related structural features of the Overthrust model without 
explicitly accounting for complex wave-propagation effects.
\
\
Reference: Aminzadeh, F., Brac, J., & Kunz, T., 1997. SEG/EAGE 3-D Salt and Overthrust Models, no. 1 in
SEG/EAGE 3-D Modeling Series, Society of Exploration Geophysicists, Tulsa, Oklahoma, Distribution
CD of Salt and Overthrust models.

\
Table S2 provides the dataset preparation details of the Opunake dataset.
### Table S2. Dataset preparation details of the Opunake dataset
| Dataset | Opunake |
|---|---|
| Initial extraction (**A**) | A 256×650×186 volume randomly selected from the full dataset |
| Training block | A 256×650×166 volume extracted from **the first 166 inlines of A** |
| Test block | A 224×224×20 volume extracted from **the last 20 inlines of A** |
| **3D dataset for full fine tuning:** | |
| Volume size | 40×40×16 |
| Stride of sliding window | 16×16×8 |
| Total volumes in the 3D collection (**B**) | 11700 |
| Training set randomly sampled from **B** | 1000 |
| Validation set randomly sampled from **B** | 250 |

\
Table S3 summarizes the parameter status of different components across the two training stages and target-survey adaptation.
### Table S3. Dataset preparation details of the Opunake dataset
| Components               | Stage 1 | Inherited from Stage 1? | Stage 2 | Opunake adaptation | Overthrust adaptation |
|--------------------------|---|---|---|---|---|
| **Transformer encoders** | Trainable | YES | Frozen | Trainable | Frozen |
| **SDAs**                 | N/A | N/A | Trainable | Trainable | Trainable |
| **Remaining components** | Trainable | NO | Trainable | Trainable | Trainable |

## S2. Settings in different components
As introduced in Sec. 3.1, the embedding channel number (or embedding dimension) is 32, 
which means the basic dimensions at the input of the convolutional layer, MSA, MLP, and SDA are all 32. 
All the convolutional layers (Conv) in T-2.5D has a channel number of 32, a kernel size of 3×3, a stride of 1, 
and a padding of 1. However, certain internal dimensions differ from this basic one. Specifically, 
the hidden dimension of the MLP is twice the embedding dimension, i.e., 64. For the SDA, 
the internal feature dimension is set to 384, with the first linear layer projecting the dimension of 
input features from 32 to 384 and the final linear layer projecting it back to 32.

## S3. About dense connections
The dense connections are implemented by concatenating features from all the previous layers at the 
channel dimension, which inevitable leads to an increase in the number of feature channels. Therefore, 
an additional Conv layer is placed after each concatenation to reduce the channel number back to 32. 
For simplicity, we omitted these layers from the architectural illustration in Fig. 4, as this dense-connection 
mechanism has been well established in the literature (Huang et al. 2017), in which more details can be found.
\
\
Reference: Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q., 2017. Densely connected convolutional networks, 
in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## S4. SSIM configuration
The SSIM we used was calculated using the _structural_similarity_index_measure_ function from _torchmetrics_ 
(a python package) with its default settings, including a Gaussian window of 11 × 11 (2D) or 11 × 11 × 11 (3D),
a standard deviation of 1.5, and constants C<sub>1</sub> = 0.01 and C<sub>2</sub> = 0.03. The data range was automatically determined 
from the predicted and reference data, and mean reduction was applied. SSIM was calculated directly in 2D for 
individual patches and in 3D for individual volumes. For sliding-window testing, SSIM was evaluated over the 
entire reconstructed test volume after reassembly, rather than averaging the SSIM values of individual patches 
or volumes. 
