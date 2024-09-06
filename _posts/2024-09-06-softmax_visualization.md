---
layout: post
title: Visualizing SoftMax
date: 2024-09-06 11:14:00+0100
description: An interactive visualization of the softmax function applied to a set of random data.
tags: softmax visualization
tag: boltzmann distribution
tag: gibbs distribution
tag: machine learning
tag: softmax visualization
categories: research
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

## The SoftMax function

The softmax function is defined as

$$  \text{softmax}( \boldsymbol{x}) = \frac{e^{x_i} }{\sum_j{e^{x_j} } } $$

It maps an $$n$$-components vector $$x \in \mathbb{R} ^ n$$ to a $$n$$ positive component vector with unitary $$L^1$$ norm, meaning all components are positive and sum to one. This effectively transforms any data collection into a probability distribution.

This property is central to classification neural networks, where it is often used in the output layer with one-hot encoded labels.

In statistical mechanics, the softmax function is used to compute the probability that a thermodynamic system occupies a particular macrostate, given the properties of all accessible microstates.

## Visualization

<iframe width="1000" height="750" frameborder="0" scrolling="yes" src="/assets/html/interactive_softmax.html"></iframe> 
