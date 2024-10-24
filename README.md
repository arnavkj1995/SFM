# Successor Feature Matching
Implementation of SFM agent introduced in the paper:

### [Non-Adversarial Inverse Reinforcement Learning via Successor Featrure Matching]() 

by [Arnav Kumar Jain](https://arnavkj1995.github.io/), [Harley Wiltzer](https://harwiltz.github.io/), [Jesse Farebrother](https://brosa.ca/), [Irina Rish](https://sites.google.com/view/irinarish/), [Glen Berseth](https://neo-x.github.io/), and [Sanjiban Choudhury](https://sanjibanc.github.io/)


## Abstract
In inverse reinforcement learning (IRL), an agent seeks to replicate expert demonstrations through interactions with the environment.
Traditionally, IRL is treated as an adversarial game, where an adversary searches over reward models, and a learner optimizes the reward through repeated RL procedures.
This game-solving approach is both computationally expensive and difficult to stabilize.
In this work, we propose a novel approach to IRL by \emph{direct policy optimization}: exploiting a linear factorization of the return as the inner product of successor features and a reward vector, we design an IRL algorithm by policy gradient descent on the gap between the learner and expert features.
Our non-adversarial method does not require learning a reward function and can be solved seamlessly with existing actor-critic RL algorithms.
Remarkably, our approach works in state-only settings without expert action labels, a setting which behavior cloning (BC) cannot solve.
Empirical results demonstrate that our method learns from as few as a single expert demonstration and achieves improved performance on various control tasks.
