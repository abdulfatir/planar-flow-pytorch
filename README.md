## Planar Flow

This repo contains a Pytorch implementation of Planar Flow presented in (Rezende and Mohamed, 2015) with experiments on a 2D density and MNIST dataset.

### 2D Density Results

![](assets/2ddensity.png)

### MNIST Results

| Model | Latent Space Size | Test Lower Bound |
:-------------------------:|:-------------------------:|:-------------------------:
VAE |  20 | -99.37 
VAE+PF (K=20) |  20 | -98.23

#### Usage

Vanilla VAE: `python vae.py`    
VAE with Planar Flow: `python vae-pf.py`


### References
(Rezende and Mohamed, 2015) Rezende, Danilo, and Shakir Mohamed. "Variational Inference with Normalizing Flows." International Conference on Machine Learning. 2015.
