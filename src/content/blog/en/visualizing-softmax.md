---
title: "Visualizing softmax"
description: "An interactive visualization of the softmax function — and why the same formula rules both neural network outputs and thermodynamic ensembles."
date: 2024-09-06
lang: en
postSlug: visualizing-softmax
math: true
---

## The softmax function

The softmax function is defined as

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

It maps an $n$-component vector $x \in \mathbb{R}^n$ to a vector with positive components and unit $L^1$ norm — all components positive, summing to one. Any collection of numbers becomes a probability distribution.

This property is central to classification neural networks, where softmax typically sits in the output layer paired with one-hot encoded labels.

The same formula has an older life in statistical mechanics: it is the **Boltzmann–Gibbs distribution**, giving the probability that a thermodynamic system occupies a state given the energies of all accessible states. What machine learning calls the "temperature" of a softmax is not a metaphor — it is literally the same parameter that physics puts in the denominator of the exponent.

## Interactive visualization

Drag the temperature and watch a random dataset reshape into sharper or flatter distributions:

<iframe width="100%" height="640" frameborder="0" scrolling="yes" src="/assets/html/interactive_softmax.html" title="Interactive softmax visualization"></iframe>
