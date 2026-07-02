---
title: "Visualizzare la softmax"
description: "Una visualizzazione interattiva della funzione softmax — e perché la stessa formula governa sia le uscite delle reti neurali sia gli ensemble termodinamici."
date: 2024-09-06
lang: it
postSlug: visualizing-softmax
math: true
---

## La funzione softmax

La funzione softmax è definita come

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Mappa un vettore a $n$ componenti $x \in \mathbb{R}^n$ in un vettore a componenti positive con norma $L^1$ unitaria — tutte le componenti positive, a somma uno. Qualsiasi collezione di numeri diventa una distribuzione di probabilità.

Questa proprietà è centrale nelle reti neurali di classificazione, dove la softmax siede tipicamente nello strato di uscita, accoppiata a etichette one-hot.

La stessa formula ha una vita più antica nella meccanica statistica: è la **distribuzione di Boltzmann–Gibbs**, che dà la probabilità che un sistema termodinamico occupi uno stato date le energie di tutti gli stati accessibili. Quella che il machine learning chiama "temperatura" della softmax non è una metafora — è letteralmente lo stesso parametro che la fisica mette al denominatore dell'esponente.

## Visualizzazione interattiva

Trascina la temperatura e osserva un dataset casuale rimodellarsi in distribuzioni più nette o più piatte:

<iframe width="100%" height="640" frameborder="0" scrolling="yes" src="/assets/html/interactive_softmax.html" title="Visualizzazione interattiva della softmax"></iframe>
