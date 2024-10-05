---
layout: distill
title: "Understanding Autoencoders: Intuition, Mathematics, and Code"
date: 2024-09-06 11:14:00+0100
description: "Unraveling the complexities of data representation through autoencoders: A journey from raw input to meaningful embeddings"
tags: autoencoder
tag: boltzmann distribution
tag: gibbs distribution
tag: machine learning
tag: autoencoder
categories: research
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

Autoencoders are a class of neural networks used for unsupervised learning. Their primary objective is to learn efficient representations of data by compressing inputs into a latent space (encoding) and then reconstructing them back to the original data (decoding). They serve as fundamental components in tasks like dimensionality reduction, denoising, and generative modeling. In this post, we will dissect the architecture and mathematics behind autoencoders, from the simplest one-layer linear autoencoder to more complex variants.

## Building AEs intuition from geometry

An autoencoder consists of two core components:

- **Encoder**: Maps an input vector $$ \mathbf{x} \in \mathbb{R}^n $$ into a latent space $$ \mathbf{z} \in \mathbb{R}^m $$ where $$ m < n $$
- **Decoder**: Reconstructs the input by mapping the latent representation $$ \mathbf{z} $$ back to the input space to produce $$ \hat{\mathbf{x} } \in \mathbb{R}^n $$

The encoder's main objective is therefore to reduce the dimensionality of the input data. It effectively works as a compressor engine: its output is a compressed representation if the original data's informations. It is clear that the larger the embedding dimension $$m$$ is, more the informations that can fit in it are. In the limiting case where the embedding's dimension is equal to original space's dimension $$m=n$$, the model can achieve perfect score (i.e. original data recovery) by simply learning the identity operation. For practical purposes, we are therefore interested in finding the correct balancing between recovery capabilities and compression ratio $$\eta = \frac{m}{n}$$ (also called compression efficiency).

Of course the AE will need to assign only the most important informations to the available degree of freedom. In other words, the model will need to find a way to extract meaningful "descriptive variables" **and** their place inside the latent space. 

The encoder, given an input point $$\mathbf{x}$$ which lives in $$\mathbb{R}^n$$, will give a "representation" of that point in the (smaller) space $$\mathbb{R}^m$$. It is therefore clear that the encoder is a mapping between the two spaces, just like a scalar field maps a point of the space into a single value. 
However, we are usually not sampling uniformly from the original space $$\mathbb{R}^n$$: in that case there are no "features" to be extracted, since **all** the possible combinations are present inside our dataset and the most efficient representation each datapoint will simply be itself.<d-footnote>I do not mean that we cannot in principle compress each datapoint separately: that would always be possible. I mean that there is no optimal way of doing a "conceptual" compression of the whole dataset, since it represent all possible datapoints.</d-footnote>

A better way of expressing this concept is to simply state that our datapoints are not randomly distributed in $$\mathbb{R}^n$$, but instead they occupy a *subspace* of it. We can be even more restrictive and claim that our data lie on a manifold, since we expect that the dofs interact in complex ways to create the our datapoints. We call this manifold the **observed manifold** $$\mathcal{M}_i$$ of our data. Therefore we have now that
\begin{equation}
\mathbf{x} \in \mathcal{M}_o \subset \mathbb{R}^n
\end{equation}

One thing that should be kept clear is that we do not know what are the "intrinsical" dof of our problem, nor the "intrinsical" relations between them: we do not have direct access to the **intrinsic manifold**. This is the exact same thing of saying that we do not know the positions and velocities of all the particles inside a volume of gas. Our data is a sampling of an (unknow) function of those (unknown) dofs, that is to say a noisy sampling of the intrinsic manifold.
It appears clear that, with the exception of drastically simplified situations, the intrisic manifold is mathematically and computationally intractable. This is not unusual in physics: it is exactly equivalent to what happens in statistical mechanics. It is impossible (and probably not useful) to directly manipulate all the phase space of (for example) a non-interacting classical gas. However, we know that we can describe a lot of the physical properties of this system by using a handful of variables like temperature, pression and volume. We can perform a mapping from a space with dimension of order exponential in particle number to a space of dimension just 3. We can *integrate out* redundant informations from the system.

Usually we are not concerned about $$\mathcal{M}_i$$, since for many practical purpose it can in principle be so extremely complex that its modeling is useless (think about what the intrinsic dofs of 1080p photos can be). We also expect that the same practical purposes can be solved by leveraging a much smaller number of descriptive dofs, exactly like in statistical mechanics. This is exactly the work of the encoder: it maps a point of the observed manifold to a point of the **latent manifold** $$\mathcal{M}_l$$. 

Let us summarize all of that:
| Concept | Description | Example |
|-|-|-|
| Observed Degree of Freedom (o-dof) | The measured variables that constitutes the values of a data point | Metereological variables collected by a some stations at a given time |
| Observed Manifold | The set of all observations together with their observed, noisy relationships | All data collected in a given time period |
| Latent Degree of Freedom (l-dof) | A minimal set of varibles capable of describing sufficiently well a datapoint | Mean temperature, pressure and moisture level |
| Latent Manifold | The manifold created acting with the encoder on each point of the observed manifold | All accessible values of mean temperature, pressure and moisture |
| Intrinsic Degree of Freedom (i-dof) | The real indipendent variables that drives the system's behaviour | Those of the Earth's atmosphere (and possibly more) |
| Intrinsic Manifold | The set of all possible states allowed by the system | All possible configurations of Earth's atmosphere |



The encoder, as we have just seen, is a mapping between manifolds. We therefore have two mappings at the moment: Intrinsic $$\to$$ Observed $$\to$$ Embedding manifolds. While one usually focuses on the second mapping, by trying to create powerful models for example, the frst mapping is also important, since if that mapping is of insufficient quality the whole model can suffer. This consideration is extremely important when one has to use such models in the real world. If the dataset is too small (insufficient i-manifold sampling) in relation to the i-manifold complexity, the model will almost surely fail to capture the most *instrinsically* important (descriptive) features, since it will focus on the most "evident" (represented, easily learnable) ones. And this is only beacuse insufficient sampling of the i-manifold will make the o-manifold **have** only the most evident features.

Let's use an example to visualize this. We want to create an embedding of the altitude profiles of patches of terrain of size 1km by 1km. We decide to sample the terrain every 100 meters, and so we obtain a set of 10 by 10 values. In this situation, the intrinsic manifold is just the surface of the planet we are measuring <d-footnote>and of all what lies on it</d-footnote>. The observed manifold is the set of all the the 100-dimensional measurements, each datapoint being the (ordered) set of the measures of the terrain. The intrisic dofs are extremely complex: think about the astronomical number of variables needed to exactly predict the altitude of every single point on Earth (including building, trees and so on)! <d-footnote>since this is an idealized situation, we are neglecting a lot practical issues, like the non-pointlike nature of measurements, which would only make learning harder</d-footnote> The observed dofs are the values of the terrain height as measured in the 100 points for each datapoint. 
The spatial variability of height over 1 square km can be well approximated, at the scale of 100m, by its mean altitude (order 0), direction and magnitude of overall slop (order 1), and higher order features like the concavity, the oblungateness and so on. If we restrict our embedding space to 5 dimensions, we can reasonably expect that our model will be able to only capture the "overall" shape of our datapoint. If we sample only flat cropfields with no buildings, or sea, our model will learn features that are subtly related with the tiny differences between datapoints. If we however sample a mountainous regiorn, or densely populated urban centers, we can reasonably expect that our model try to reproduce some of the pattern found in the dataset. How crucial it is a correct data sampling can be illustrated by thinking to what happens if all of our mountain samples are taken centered on the smmit. The model can learn to give a sense to different types of peaks this way; however, if a sample from the mountain wall is fed inside the model its representation will most prabably be unmeaningful.

An important note: in the example above, the plain is **flat** in two different ways: 
1) all the datapoints share the same basic structure (those of a flat plane plus minor corrections)
2) all the observed dofs are close approximations of the underlying intrinsic dofs (because there is not a great variability between the physical points on the measured surface); in other words by changing the sampling on the terrain from a regular grid to another distribution would not change much the resulting representation vector

It would be possible to achieve flatness "of the first kind" even for mountain regions: all that suffices is to restrict the i-manifold to the surroundings of the peaks: then all of the datapoints will describe similar bahviours. However, in this case it is not guaranteed a flatness "of the second kind". It could be in principle achieved by incrementing the number of points to be samples, that is to say to increase the dimensionality of the o-manifold.  

To summarize what we saw in this example:
- the more locally-rough the i-manifold is, the higher the o-manifold dimensions is required to correctly represent it
- the more globally-diverse the i-manifold is, the higher the number of samples are needed to capture all the possible sub-structures (i.e. to represent almost equally well all datapoints)
- as stated at the beginning of this post, the higher the dimensionality of the embedding space the more the features that can be learned: we will be able to extract more insights about the the i-manifold!

In fact, one of the key objectives of dimensiobnality reduction is to understand more easily the i-manifold. Assuming a sufficiently expressive o-manifold, the e-manifold (emebdding manifold) will provide a "compactified" version of the i-manifold! By compactified here I mean "reduced in size" (with fewer dofs). The encoder effectively integrates out the redundant, non-descriptive dofs in favour of the most important descriptors of every datapoint. Just as we humans can invent a categorization of volcanos by looking at the *overall* geometry of their cones, an encoder can will give numerical values that will be needed to recreate as close as possible the original datapoint, but starting from a much more compressed set of values. 


### Linear Autoencoder

The simplest form of an autoencoder is linear, where both the encoder and decoder are linear mappings. The encoder is represented as a linear transformation:

\begin{equation}
\mathbf{z} = \mathbf{A} \mathbf{x}, \quad \mathbf{A} \in \mathbb{R}^{m \times n}
\end{equation}

The decoder reconstructs the original input by applying the transpose of $$ \mathbf{A} $$:

\begin{equation}
\hat{\mathbf{x} } = \mathbf{A}^T \mathbf{z} = \mathbf{A}^T \mathbf{A} \mathbf{x}
\end{equation}

Thus, the model learns a projection onto the subspace defined by the columns of $$ \mathbf{A} $$, where $$ \mathbf{A} $$ acts as the encoder and $$ \mathbf{A}^T $$ as the decoder.

If the input data $$ \mathbf{x} $$ is centered (i.e., zero-mean), this linear autoencoder performs similarly to **Principal Component Analysis (PCA)**.

#### PCA and Linear Autoencoders: A Mathematical Perspective

Let us briefly sketch why a linear autoencoder is equivalent to PCA. In PCA, we seek to find the principal components by maximizing the variance of the projected data while minimizing the reconstruction error in a low-dimensional subspace.

**PCA Problem:**
PCA aims to find a matrix $$ \mathbf{A} \in \mathbb{R}^{m \times n} $$ that solves:

\begin{equation}
\min_{ \mathbf{A} } \| \mathbf{x} - \mathbf{A} ^ T \mathbf{A} \mathbf{x} \|^2
\end{equation}

This is precisely the objective of a linear autoencoder, where the encoder performs a projection onto the lower-dimensional space spanned by the columns of $$ \mathbf{A} $$, and the decoder projects back into the original space.

<div style="border:1px solid #ccc; padding:10px;">
<b>Formal Proof Sketch (PCA and Linear Autoencoder Equivalence):</b>  
1. Assume the input data is centered, i.e., $$ \mathbb{E}[\mathbf{x}] = 0 $$.  
2. The linear encoder projects data: $$ \mathbf{z} = \mathbf{A} \mathbf{x} $$.  
3. The linear decoder reconstructs: $$ \hat{\mathbf{x} } = \mathbf{A}^T \mathbf{z} = \mathbf{A}^T \mathbf{A} \mathbf{x} $$.  
4. The reconstruction error is:  
   \begin{equation}
   \mathcal{L}(\mathbf{x}, \hat{\mathbf{x} }) = \|\mathbf{x} - \mathbf{A}^T \mathbf{A} \mathbf{x}\|^2
   \end{equation}  
5. PCA aims to minimize this error by selecting the subspace spanned by the leading principal components of $$ \mathbf{x} $$, which is equivalent to optimizing the autoencoder weights $$ \mathbf{A} $$.
</div>

### Loss Function

The typical loss function for an autoencoder is the Mean Squared Error (MSE), which quantifies the difference between the input $$ \mathbf{x} $$ and its reconstruction $$ \hat{\mathbf{x} } $$:

\begin{equation}
L(\mathbf{x}, \hat{\mathbf{x} }) = \|\mathbf{x} - \hat{\mathbf{x} }\|^2 = \|\mathbf{x} - \mathbf{A}^T \mathbf{A} \mathbf{x}\|^2
\end{equation}

### Code Example: Linear Autoencoder in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple Linear Autoencoder
class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = torch.matmul(self.encoder.weight.t(), z)
        return x_hat

# Model, loss, optimizer setup
input_dim = 784  # Example: flattened 28x28 image (e.g., from MNIST)
latent_dim = 64
model = LinearAutoencoder(input_dim, latent_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Sample training loop
for epoch in range(100):
    for data in dataloader:
        inputs = data.view(-1, input_dim)
        outputs = model(inputs)

        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This code demonstrates a basic linear autoencoder using PyTorch, where the encoder and decoder are simple linear layers.

## Non-linear Autoencoder

Linear autoencoders, as shown, are limited to capturing linear relationships within the data. To capture more complex structures, non-linear activation functions (e.g., ReLU, Sigmoid) are introduced between the layers.

For a single-layer non-linear autoencoder, the encoder becomes:

\begin{equation}
\mathbf{z} = f(\mathbf{A} \mathbf{x} + \mathbf{b})
\end{equation}

where $$ f $$ is a non-linear activation function such as ReLU, and $$ \mathbf{b} $$ is a bias term. Similarly, the decoder becomes:

\begin{equation}
\hat{\mathbf{x} } = g(\mathbf{A'} \mathbf{z} + \mathbf{b'})
\end{equation}

where $$ g $$ is another non-linearity, and $$ \mathbf{A'} \in \mathbb{R}^{n \times m} $$ is the decoder weight matrix.

### Code Example: Non-linear Autoencoder in PyTorch

```python
class NonlinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NonlinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

This non-linear autoencoder uses ReLU activations between its layers to capture more intricate relationships.

## Undercomplete vs. Overcomplete Autoencoders

- **Undercomplete Autoencoder**: The latent space $$ \mathbb{R}^m $$ has lower dimensionality than the input space $$ \mathbb{R}^n $$, i.e., $$ m < n $$. This encourages the model to learn only the most essential features.
  
- **Overcomplete Autoencoder**: The latent space has a higher dimensionality than the input space $$ \mathbb{R}^n $$, i.e., $$ m > n $$. Without regularization, this model might learn the identity function. Therefore, regularization methods such as sparsity constraints or adding noise are critical in these models.

## Regularization Techniques

### Denoising Autoencoder (DAE)

A Denoising Autoencoder aims to reconstruct clean data from corrupted inputs. The input $$ \mathbf{x} $$ is corrupted by adding noise:

\begin{equation}
\tilde{\mathbf{x} } = \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\end{equation}

The loss function remains the MSE between the reconstructed $$ \hat{\mathbf{x} } $$ and the clean $$ \mathbf{x} $$.

### Sparsity Regularization

Sparse autoencoders encourage the model to learn a sparse representation by penalizing the activation of the hidden units. This can be achieved by adding a regularization term, such as the Kullback-Leibler (KL) divergence between the average activation of each latent unit and a target sparsity level $$ \rho $$:

\begin{equation}
L_{\text{KL} }(\rho || \hat{\rho}) = \sum_{i=1}^{m} \rho \log \frac{\rho} {\hat{\rho}_i} + (1-\rho) \log \frac{1-\rho} {1-\hat{\rho}_i}
\end{equation}

The overall loss becomes:

\begin{equation}
L_{ \text{total} } = L(\mathbf{x}, \hat{\mathbf{x} }) + \lambda L_{\text{KL} }
\end{equation}

where $$ \lambda $$ controls the strength of the sparsity penalty.

## Applications of Autoencoders

- **Dimensionality Reduction**: Similar to PCA, autoencoders can reduce the dimensionality of data while preserving important features. This is useful for visualization or as a preprocessing step for other tasks like classification.
  
- **Denoising**: Denoising Autoencoders (DAEs) are effective for removing noise from data, especially images.
  
- **Anomaly Detection**: By training an autoencoder on normal data, it can detect anomalies during inference, as anomalies typically have a higher reconstruction error.

- **Generative Models**: Variational Autoencoders (VAEs) are used to generate new data points by learning a probabilistic model of the latent space.

## Conclusion

Autoencoders are a powerful and versatile tool for learning representations in an unsupervised manner. By understanding their architecture and mathematical foundation, we can apply them to a variety of tasks, from data compression to generative modeling.

