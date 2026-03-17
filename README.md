
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
---

## Method

### 1. Subset Evaluation

Given an input with  n (n=|S|)features (e.g., token, words or maybe sentences), we evaluate the model on all subsets:

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

- reward2Iandmat:A matrix of storage coefficients $(-1)^{|S|-|L|}$ （Implemented in and_or_harsanyi_utils.py）
- Reward functions (For classification models, the reward is the logits of the correct category after applying softmax.) (I implement this in and_or_harsanyi.py)
- EP: a index defined as $EP(S) = |v(s)-v(\phi)|/|v(N)-v(\phi)|$

These interactions capture **non-linear dependencies** between features beyond individual contributions.

---

### 3. Interaction Graph

We construct an **Interaction Decision Graph**:

- Nodes: input features (tokens)
- Edges: interaction strength (I(a,b))
- Communities: discovered via clustering (use Louvain algorithm)

This allows us to interpret model reasoning structure.

---

## Experiments

We apply the framework to:

- **Dataset**: GSM8K (math reasoning)
- **Model type**: generative LLMs (gpt-oss)
  
### Experimental process:

- 使用用gpt-oss-20b进行生成式任务，不再为gsm8k的prompt添加选项。

- 注意到生成式任务的答案token不一定在固定位置出现，所以我们采取遍历生成的所有token的正确答案的logits，取最大值并log(p/1-p)后作为全集的reward, get_reward函数要重新写。

- 需要修改生成mask_batch的逻辑，改为生成式任务后，每个mask_batch都需要生成完整文本（要遍历token获得正确答案最大的logits），不能只调用一次forward函数

- label文件改为正确答案tokenizer后的token_id,作为之后取logits的索引

- generate()函数只接受input_ids, 但我们希望在embedding层进行mask，改为手写采forward loop的方式，堆叠每步生成的logits，并取最大值

- 把计算交互的baseline改为v(N)后，感觉整体交互变强，比如原来baseline为v(\phi)时某些涵盖重要条件的交互组合的交互值接近0或者为负数，但是baseline为v(N)时，这些交互值大幅提高（但是也会出现不重要信息交互值过高的情况）

- 改变baseline后，交互最强的组合不再是“最后一步的推理+ Final answer:”，而是题目中的关键信息作为交互最强的组合，感觉使用$I(a,b) = v(N) - v(N/a) + v(N/b) + v(a,b)$在gsm8k数据集上比$I(a,b) = v(a,b) - v(a) - v(b) + v(\phi)$可能更优
 
### Thoughts on Classification Tasks

- 如果要进行判别式任务，应该选择在多项选择题数据集上微调过的模型,（现在看来deberta即使在qsac上微调过也会受无关选项影响），模型会觉得选项也是文本特征的一部分导致误判（选项也参与了attention分布）
  
- 在qsac数据集的实验结果(路径：results/qsac)，sample1的intersection结果I(2,9)值特别高，而含fact1和问题的交互值很低，这是不合理的，没有fact1无法推论出“病毒会导致不正常的细胞生长”，没有问题模型为什么笃定选G？合理怀疑模型只是根据fact2选择G，(微调是过拟合的，只是保证高准确率),模型并没有进行思维链行为。




### Goals

- Analyze whether models follow structured reasoning ("Question->fact1 ->fact 2->answer”)
  
### Results

- You can check visual_communities and intersection values in results document

  
