# Interaction-Based Interpretability for LLM Reasoning

## Overview

This project implements an interaction-based interpretability framework for analyzing decision-making processes in neural networks, with a focus on large language models (LLMs).

The core idea is to model feature contributions using **Harsanyi interactions** and **Shapley-style decompositions**, enabling the analysis of **higher-order interactions** between input features (e.g., tokens).

We further apply this framework to study whether LLMs follow structured reasoning patterns in tasks such as GSM8K.

---

## Key Features

- Implementation of **Harsanyi interactions** for neural networks  
- Support for **Shapley value** and **Shapley interaction index**  
- Efficient computation of model outputs over **all feature subsets**  
- Construction of **interaction graphs** for interpretability  
- Extension to **generative models (LLMs)** using sequence-level rewards  
- Optimization-based feature selection inspired by **Meaningful Perturbation**  

---

## Method

### 1. Subset Evaluation

Given an input with \( n \) features (e.g., tokens), we evaluate the model on all subsets:

\[
v(S) = f(x_S)
\]

where \( S \subseteq N \), and \( x_S \) is the masked input.

---

### 2. Interaction Computation

We compute higher-order feature interactions using:

- Harsanyi interaction
- Shapley interaction index

These interactions capture **non-linear dependencies** between features beyond individual contributions.

---

### 3. Interaction Graph

We construct an **Interaction Decision Graph**:

- Nodes: input features (tokens)
- Edges: interaction strength
- Communities: discovered via clustering (e.g., Louvain)

This allows us to interpret model reasoning structure.

---

### 4. Extension to LLMs

For generative models, we define a **reward function** based on sequence probabilities:

\[
R(S) = \log P(y \mid x_S)
\]

This enables analysis of:

- Chain-of-Thought reasoning
- Dependency between question, intermediate steps, and final answer

---

### 5. Optimization-Based Feature Selection

We implement a method inspired by **Meaningful Perturbation**:

- Learn a sparse mask over input features  
- Maximize model response under constraints  
- Identify the most influential subset of tokens  

---

## Experiments

We apply the framework to:

- **Dataset**: GSM8K (math reasoning)
- **Model type**: generative LLMs

### Goals

- Analyze whether models follow structured reasoning:
  
