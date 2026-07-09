---
title: "The Boris algorithm: simulating relativistic particles in electromagnetic fields"
description: "Why the motion of charged particles resists analytical solutions, and how the Boris pusher integrates it with remarkable accuracy — with open-source C++ experiments."
date: 2024-09-04
lang: en
postSlug: boris-algorithm
math: true
---

## The problem

The motion of charged relativistic particles interacting with an external electromagnetic (EM) field is governed by the Lorentz force (here in SI units):

$$
\frac{d(\gamma\mathbf{v})}{dt} = \alpha \left( \mathbf{E} + \mathbf{v}\times \mathbf{B} \right)
$$

where $\gamma = (1 - v^2/c^2)^{-1/2}$ is the Lorentz factor and $\alpha = q/m$, with $q$ and $m$ the particle's charge and mass. In the non-relativistic limit the left-hand side reduces to

$$
\frac{d\mathbf{v}}{dt} = \alpha \left( \mathbf{E} + \mathbf{v}\times \mathbf{B} \right)
$$

which is a linear first-order differential equation.

For some field configurations the trajectory can be found analytically. But those scenarios are abstractions of the far more complex field geometries found in nature and technology, and the resulting solutions are only qualitatively correct. Worse: moving charges generate their own fields, which perturb the paths of other particles, which in turn regenerate the fields — microscopic motion becomes chaotic and analytically hopeless for any practical situation.

If only the *collective* motion of a large enough population matters, statistical mechanics takes over (that road leads to plasma physics). But if we care about the motion of individual particles, we must integrate the equation numerically. That is what this work does.

## Why it matters

Charged-particle dynamics sits at the core of several areas of modern physics:

- **Astrophysics** — from the solar corona to ultra-relativistic electrons and ions gyrating in the galactic magnetic field, charged particles are everywhere in the cosmos, and their motion is a probe of both stellar objects and cosmic evolution.
- **Particle accelerators** — creating and maintaining coherent packets of highly relativistic particles requires exquisitely accurate computation of trajectories in complex magnetic fields.
- **Controlled nuclear fusion** — manipulating high-density, high-temperature plasma in strong non-uniform magnetic fields is among the hardest control problems ever attempted, and simulation is half the battle.

## The method

We ignore particle-generated fields and particle–particle interactions, and concentrate on motion in *external* EM fields. The classical equation is easy to integrate; real scenarios, however, involve at least mildly relativistic particles. So we solve the relativistic equation in both its classical and ultra-relativistic limits using the **Boris algorithm** — the de-facto standard pusher in astrophysical and plasma simulation, prized for its long-term stability: it conserves phase-space volume, so energy errors stay bounded over millions of timesteps where naive integrators drift.

## The experiments

The full C++ implementation — Boris pusher plus an RK4 comparison, validated against analytical solutions with convergence analysis — is open source, organized as a set of reproducible experiments: gyration, E×B drift, magnetic dipole trapping, magnetic bottles, X-point reconnection.

<video autoplay loop muted playsinline preload="metadata" aria-label="Charged particle trajectories near an X-point magnetic reconnection region"><source src="/assets/video/xpoint_animation.mp4" type="video/mp4" /></video>

<video autoplay loop muted playsinline preload="metadata" aria-label="Particle trapped in a magnetic dipole field — a miniature radiation belt"><source src="/assets/video/dipole_animation.mp4" type="video/mp4" /></video>

Code, experiments and the full write-up: [ChargedParticleSimulator on GitHub](https://github.com/luca-bottero/ChargedParticleSimulator).
