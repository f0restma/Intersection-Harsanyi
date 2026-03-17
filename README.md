<img width="3877" height="531" alt="image" src="https://github.com/user-attachments/assets/cacad7e5-3f2a-4533-bb91-bc3f30ac235e" /># Interaction-Based Interpretability for LLM Reasoning

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

$$
v(S) = \sum_{L \subseteq S\}v(L)
$$

where  $S \subseteq N \$, and \( L \) is the masked input.

As we know, Harsanyi dividend:

$$
I^{Harsanyi}(S) = \sum_{L \subseteq S}(-1)^{|S|-|L|}v(L)
$$

The define of two order Harsanyi-intersection  in our project:

$I(a,b) = v(a,b) - v(a) - v(b) + v(\phi)$

Change the baseline from the from $\phi$ to N, we get:

$I(a,b) = v(N) - v(N/a) + v(N/b) + v(a,b)$ (I implement this in and_or_harsanyi_utils.py)

---

### 2. Interaction Computation

We compute higher-order feature interactions using:

- A matrix of storage coefficients $(-1)^{|S|-|L|}$ （Implemented in and_or_harsanyi_utils.py）
- Reward functions (For classification models, the reward is the logits of the correct category after applying softmax.) (I implement this in and_or_harsanyi.py)

These interactions capture **non-linear dependencies** between features beyond individual contributions.

---

### 3. Interaction Graph

We construct an **Interaction Decision Graph**:

- Nodes: input features (tokens)
- Edges: interaction strength (I(a,b))
- Communities: discovered via clustering (use Louvain algorithm)

This allows us to interpret model reasoning structure.

---

### 4. Extension to LLMs

For generative models, we define a **reward function** :

$$
R(S) = softmax(logits(answer_token))
$$



---


## Experiments

We apply the framework to:

- **Dataset**: GSM8K (math reasoning)
- **Model type**: generative LLMs (gpt-oss)
  
Experimental process:
-使用用gpt-oss-20b进行生成式任务，不再为gsm8k的prompt添加选项。

-注意到生成式任务的答案token不一定在固定位置出现，所以我们采取遍历生成的所有token的正确答案的logits，取最大值并log(p/1-p)后作为全集的reward, get_reward函数要重新写。

-需要修改生成mask_batch的逻辑，改为生成式任务后，每个mask_batch都需要生成完整文本（要遍历token获得正确答案最大的logits），不能只调用一次forward函数

-label文件改为正确答案tokenizer后的token_id,作为之后取logits的索引

-generate()函数只接受input_ids, 但我们希望在embedding层进行mask，改为手写采forward loop的方式，堆叠每步生成的logits，并取最大值

-对于较复杂的题目（需要结合多个信息才能做出正确解答），聚类的结果显示模型往往把含有最终答案和“Final answer”放在一起，这说明模型可能没有进行推理过程，而是“偷看”答案才回答正确，并认为直接给出答案的句子很重要

### Goals

- Analyze whether models follow structured reasoning ("Question->fact1 ->fact 2->answer”)
  
