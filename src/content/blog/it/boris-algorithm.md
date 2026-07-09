---
title: "L'algoritmo di Boris: simulare particelle relativistiche nei campi elettromagnetici"
description: "Perché il moto delle particelle cariche resiste alle soluzioni analitiche, e come l'integratore di Boris lo simula con notevole accuratezza — con esperimenti C++ open source."
date: 2024-09-04
lang: it
postSlug: boris-algorithm
math: true
---

## Il problema

Il moto di particelle cariche relativistiche che interagiscono con un campo elettromagnetico (EM) esterno è governato dalla forza di Lorentz (qui in unità SI):

$$
\frac{d(\gamma\mathbf{v})}{dt} = \alpha \left( \mathbf{E} + \mathbf{v}\times \mathbf{B} \right)
$$

dove $\gamma = (1 - v^2/c^2)^{-1/2}$ è il fattore di Lorentz e $\alpha = q/m$, con $q$ ed $m$ carica e massa della particella. Nel limite non relativistico il primo membro si riduce a

$$
\frac{d\mathbf{v}}{dt} = \alpha \left( \mathbf{E} + \mathbf{v}\times \mathbf{B} \right)
$$

che è un'equazione differenziale lineare del primo ordine.

Per alcune configurazioni di campo la traiettoria si trova analiticamente. Ma quegli scenari sono astrazioni di geometrie di campo ben più complesse, quelle che si incontrano in natura e nella tecnologia, e le soluzioni che ne derivano sono corrette solo qualitativamente. C'è di peggio: le cariche in moto generano a loro volta campi, che perturbano le traiettorie delle altre particelle, che a loro volta rigenerano i campi — il moto microscopico diventa caotico e analiticamente intrattabile in ogni situazione pratica.

Se conta solo il moto *collettivo* di una popolazione abbastanza grande, subentra la meccanica statistica (quella strada porta alla fisica del plasma). Ma se ci interessa il moto delle singole particelle, dobbiamo integrare l'equazione numericamente. Ed è esattamente ciò che facciamo qui.

## Perché è importante

La dinamica delle particelle cariche è al cuore di diverse aree della fisica moderna:

- **Astrofisica** — dalla corona solare agli elettroni e ioni ultra-relativistici che spiraleggiano nel campo magnetico galattico: le particelle cariche sono ovunque nel cosmo, e il loro moto è una sonda tanto degli oggetti stellari quanto dell'evoluzione cosmica.
- **Acceleratori di particelle** — creare e mantenere pacchetti coerenti di particelle altamente relativistiche richiede un calcolo estremamente accurato delle traiettorie in campi magnetici complessi.
- **Fusione nucleare controllata** — manipolare plasma ad alta densità e temperatura in campi magnetici intensi e non uniformi è tra i problemi di controllo più difficili mai affrontati, e la simulazione è metà della battaglia.

## Il metodo

Ignoriamo i campi generati dalle particelle e le interazioni particella-particella, concentrandoci sul moto in campi EM *esterni*. L'equazione classica è facile da integrare; gli scenari reali, però, coinvolgono particelle almeno moderatamente relativistiche. Risolviamo quindi l'equazione relativistica nei suoi limiti classico e ultra-relativistico con l'**algoritmo di Boris** — lo standard di fatto nelle simulazioni astrofisiche e di plasma, apprezzato per la stabilità sul lungo periodo: conserva il volume dello spazio delle fasi, così gli errori sull'energia restano limitati su milioni di passi temporali, dove gli integratori ingenui derivano.

## Gli esperimenti

L'implementazione C++ completa — integratore di Boris più un confronto RK4, validata contro soluzioni analitiche con analisi di convergenza — è open source, organizzata come una serie di esperimenti riproducibili: girazione, deriva E×B, intrappolamento in dipolo magnetico, bottiglie magnetiche, riconnessione a punto X.

<video autoplay loop muted playsinline preload="metadata" aria-label="Traiettorie di particelle cariche vicino a una regione di riconnessione magnetica a punto X"><source src="/assets/video/xpoint_animation.mp4" type="video/mp4" /></video>

<video autoplay loop muted playsinline preload="metadata" aria-label="Particella intrappolata in un campo di dipolo magnetico — una fascia di radiazione in miniatura"><source src="/assets/video/dipole_animation.mp4" type="video/mp4" /></video>

Codice, esperimenti e relazione completa: [ChargedParticleSimulator su GitHub](https://github.com/luca-bottero/ChargedParticleSimulator).
