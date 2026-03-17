import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Union, Iterable, List, Tuple, Callable, Type, Dict
import torch.nn.functional as F
from .and_or_harsanyi_utils import (get_reward2Iand_mat, get_reward2Ior_mat, get_reward2Ishapley_mat, 
                                   get_reward2Ishapley_interaction_mat)
from .reward_function import get_reward
from .plot import plot_simple_line_chart, plot_interaction_progress,plot_multi_line_chart
from .set_utils import flatten, generate_all_masks, generate_all_masks_re, generate_all_communities
from tqdm import tqdm
import torch.optim as optim
from utils import huber
from math import comb

def l1_on_given_dim(vector: torch.Tensor, indices: List) -> torch.Tensor:
    assert len(vector.shape) == 1
    strength = torch.abs(vector)
    return torch.sum(strength[indices])

def torch_comb_safe(n: torch.Tensor, k: int) -> torch.Tensor:
    # 计算 log(C(n, k)) = log(n!) - log(k!) - log((n-k)!))
    log_n = torch.lgamma(n.float() + 1)  # log(n!)
    log_k = torch.lgamma(torch.tensor(k + 1.0))  # log(k!)
    log_nk = torch.lgamma(n.float() - k + 1)  # log((n-k)!)
    log_comb = log_n - log_k - log_nk
    return torch.exp(log_comb).to(n.dtype)

def generate_ckpt_id_list(niters: int, nckpt: int) -> List:
    ckpt_id_list = list(range(niters))[::max(1, niters // nckpt)]
    # force the last iteration to be a checkpoint
    if niters - 1 not in ckpt_id_list:
        ckpt_id_list.append(niters - 1)
    return ckpt_id_list


class AndOrHarsanyi(object):
    def __init__(
        self,
        forward_function: Union[Type[nn.Module], Callable],
        selected_dim: Union[None, str],
        x: torch.Tensor,
        baseline: Union[torch.Tensor, int], # for nlp, this baseline can be an integer baseline flag
        y: int,
        sample_id: int,
        all_players_subset: Union[None, tuple, list] = None,
            # sometimes we just choose a subset of input variables as the set of all players N.
            # i.e., we have customized players
            # Example: for NLP, we can choose a subset of words (may contain several tokens) as the set of all players N
            # such as all_players_subset = [[0], [2, 3], [5], [7, 8, 9]]
        background: Union[None, tuple, list] = None,
        background_type: str = "ori",
        mask_input_function: Callable = None,
        cal_batch_size: int = None,   # batch size for computing forward passes on masked input samples
        softmax_sample_dims = None,
        sort_type: str = "order",
        verbose: int = 1,
        calculator: Callable = None,
        interaction_type = None
    ):
        assert x.shape[0] == 1, "Only support batch size 1"
        assert sort_type in ["order", "binary"]
        assert background_type in ["ori", "mask"]

        if isinstance(baseline, torch.Tensor):
            assert baseline.shape[0] == 1
        self.forward_function = forward_function
        self.selected_dim = selected_dim
        self.input = x
        self.target = y
        self.baseline = baseline
        self.sort_type = sort_type
        self.softmax_sample_dims = softmax_sample_dims
        self.verbose = verbose
        self.sample_id = sample_id
        self.device = x.device
        self.calculator = calculator
        self.interaction_type = interaction_type

        if background is None:
            background = []
        self.background = background  # players that always exists / absent (default: emptyset []), depending on background_type
        self.background_type = background_type

        self.mask_input_function = mask_input_function  # for different data type (image, text, etc.), the mask_input_function can be different
        self.cal_batch_size = cal_batch_size

        self.n_input_variables = self.input.shape[1]
        self.all_players_subset = all_players_subset  # customized players
        if all_players_subset is not None:
            self.n_players = len(all_players_subset)
        else:
            self.n_players = self.n_input_variables

        if self.interaction_type in ["harsanyi", "shapley_taylor", "shapley_interaction_index", "shapley", "shapleyC"]:
            print("Generating player masks...")
            self.player_masks = torch.BoolTensor(generate_all_masks(length=self.n_players,
                                                                    sort_type=self.sort_type,
                                                                    #k=2
                                                                    )).to(self.device)
            # self.player_masks shape: (2 ** n_players, n_players)
            print("done")

            self.reward2Iand = get_reward2Iand_mat(n_dim=self.n_players, sort_type=self.sort_type,reverse=False).to(self.device)
            #self.reward2Ior = get_reward2Ior_mat(n_dim=self.n_players, sort_type=self.sort_type).to(self.device)
        else:
            self.player_masks = None
            self.reward2Iand = None

        """
        Difference between player_mask and sample_mask:
        - player_masks: each mask is of length (n_players,), the mask does not includes any background variables
            One player can correspond to several tokens in NLP tasks
        - sample_masks: each mask is of length (n_input_variables,), the mask includes both players and background variables
        Example:
            input_ids = [[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]]
            all_players_subset = [[0], [2, 3], [7, 8, 9]] -> [0,2,3,7,8,9]
            And assume background_type = "ori"

          Then we have: n_input_variables = 10, n_players = 3
            player_masks = [[False, False, False],
                            [ True, False, False],
                            [False,  True, False],
                            [False, False,  True],
                            [ True,  True, False],
                            [ True, False,  True],
                            [False,  True,  True],
                            [ True,  True,  True]]
            sample_masks = [[False,  True, False, False,  True,  True,  True, False, False, False],
                            [ True,  True, False, False,  True,  True,  True, False, False, False],
                            [False,  True,  True,  True,  True,  True,  True, False, False, False],
                            [False,  True, False, False,  True,  True,  True,  True,  True,  True],
                            [ True,  True,  True,  True,  True,  True,  True, False, False, False],
                            [ True,  True, False, False,  True,  True,  True,  True,  True,  True],
                            [False,  True,  True,  True,  True,  True,  True,  True,  True,  True],
                            [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]
        """
    def calculate_all_subset_rewards(self): # modify self.player_masks, self.sample_masks, self.S_list, self.rewards
        if self.all_players_subset is None: # it means all input variables are considered as players, and there are no background variables
            assert len(self.background) == 0 and self.mask_input_function is None
            self.sample_masks = self.player_masks  # bool tensor

        else: # only a subset of input variables are considered as players, and the other variables are considered as background variables
            assert self.background is not None and self.mask_input_function is not None
            all_players_subset_arr = self.all_players_subset
            if self.interaction_type == "opt":
                # 使用 soft player mask，映射到 input token 空间，保持梯度
                full_masks = []
                for i in range(self.player_masks.shape[0]):
                    full_mask = torch.zeros(self.n_input_variables, device=self.device)
                    for j, token_indices in enumerate(all_players_subset_arr):
                        full_mask[token_indices] = self.player_masks[i, j]
                    full_masks.append(full_mask)
                self.sample_masks = torch.stack(full_masks, dim=0)  # float tensor, [batch, n_input_variables]
            else:
                all_players_subset_arr = np.array(self.all_players_subset, dtype=object)
                self.sample_masks = []
                for i in tqdm(range(self.player_masks.shape[0]), ncols=100, desc="Generating sample masks"):
                # for i in range(self.player_masks.shape[0]):
                    player_mask = self.player_masks[i].clone().cpu().numpy() # bool tensor -> bool array
                    # print(f"\nplayer_mask: {player_mask}")
                    sample_mask = np.zeros(self.n_input_variables, dtype=bool)

                    if self.background_type == "ori":
                        sample_mask[flatten([self.background, all_players_subset_arr[player_mask]])] = True
                    elif self.background_type == "mask":
                        sample_mask[flatten([all_players_subset_arr[player_mask]])] = True
                    else:
                        raise NotImplementedError(f"Invalid background type: {self.background_type}")
                    self.sample_masks.append(sample_mask)
                    #print(f"sample_mask: {sample_mask}")
                self.sample_masks = np.stack(self.sample_masks, axis=0)
                self.sample_masks = torch.BoolTensor(self.sample_masks).to(self.device)
        if self.cal_batch_size is None:
            self.cal_batch_size = self.player_masks.shape[0]

        rewards = []

        if self.interaction_type in ("opt", "hard"):
            pbar = range(int(np.ceil(self.player_masks.shape[0] / self.cal_batch_size)))
        else:
            if self.verbose:
                pbar = tqdm(range(int(np.ceil(self.player_masks.shape[0] / self.cal_batch_size))), ncols=100, desc="Calc model outputs")
            else:
                pbar = range(int(np.ceil(self.player_masks.shape[0] / self.cal_batch_size)))

        for batch_idx in pbar:
            sample_mask_batch = self.sample_masks[batch_idx * self.cal_batch_size: (batch_idx + 1) * self.cal_batch_size]
            # print(f"\nsample_mask_batch:{sample_mask_batch}\n")
            masked_inputs_batch = self.mask_input_function(self.input, self.baseline, sample_mask_batch)
            # print(f"masked_inputs_batch:{masked_inputs_batch}")
            # masked_inputs_batch = masked_inputs_batch[:, useful_loc] # todo: 这是干什么的？
            if self.interaction_type == "opt":
                attention_mask = (sample_mask_batch > 0.1).long()  # 用一个阈值构造 soft->hard 的 attention_mask
                output_batch = self.forward_function(inputs_embeds=masked_inputs_batch,
                                                     attention_mask=attention_mask)
                # output_batch = self.forward_function(inputs_embeds=masked_inputs_batch)
                # --- 新增处理逻辑 ---
                # 如果输出是 [Batch, 2]，我们需要根据目标类别提取对应的维度
                if output_batch.dim() == 2 and output_batch.shape[1] > 1:
                    # 这里的 self.target 应该是 0 或 1
                    target_idx = self.target if isinstance(self.target, int) else self.target[0]
                    output_batch = output_batch[:, target_idx] 
                # -------------------

            else:
            #     output_batch = self.forward_function(masked_inputs_batch)
                print(masked_inputs_batch)
                #使用 forward_function 自己实现 generate 循环
                max_new_tokens = 8 
                batch_size = masked_inputs_batch.shape[0] 
                logits_steps = [] # 保存每步 logits（替代 generate().scores）
                for t in range(max_new_tokens):
                    scores = self.forward_function(masked_inputs_batch)# 调用forward_function  scores.logits 形状：[B, L, vocab]
                    logits = scores[:, -1, :] # 取最后一个 token 的 logits → [B, vocab]
                    logits_steps.append(logits)
                    # 选下一个 token（与 generate 行为保持一致）
                    next_token = logits.argmax(dim=-1) # [B]
                    # 拼接到序列末尾
                    masked_inputs_batch = torch.cat([masked_inputs_batch, next_token.unsqueeze(1)], dim=1)
                    # 堆叠 logits → 得到 [num_steps, batch_size, vocab_size]
                    logits_all_steps = torch.stack(logits_steps, dim=0)
                    # 兼容原来的逻辑（从此处开始完全不变）
                    vocab_size = logits_all_steps.shape[2]
                    # 准备 target token id
                if isinstance(self.target, int): 
                    y_ids = [self.target] 
                else: 
                    y_ids = self.target
                y_ids = [id_ for id_ in y_ids if id_ < vocab_size] 
                if len(y_ids) == 0: 
                    raise ValueError("All target token_ids are out of vocab!")
                # 计算 log_softmax
                log_probs = torch.nn.functional.log_softmax(logits_all_steps, dim=2)
                # 取目标 token 的概率
                log_probs_for_y = log_probs[:, :, y_ids] # [steps, B, #tokens] 
                print(log_probs_for_y)
                # sum over token dimension
                joint_log_probs = log_probs_for_y.sum(dim=2) # [steps, B] 
                joint_log_probs = torch.clamp(joint_log_probs, min=-5.0)
                print(joint_log_probs)
                # 找出每个 sample 的最佳 step
                best_step_idx = joint_log_probs.argmax(dim=0) # [B] 
                batch_idx = torch.arange(joint_log_probs.shape[1], device=joint_log_probs.device) 
                best_joint_log_prob = joint_log_probs[best_step_idx, batch_idx] # [B]
                # 转成概率
                p = torch.exp(best_joint_log_prob).float()
                # reward 输出 
                output_batch = torch.log(p / (1 - p + 1e-10))
            # todo: 20250118 改成了每个Batch都计算reward，这样可以减小显存占用？ —— 可以
            # reward_batch = get_reward(output_batch,
            #                           selected_dim=self.selected_dim,
            #                           gt=self.target,
            #                           sample=self.softmax_sample_dims)
            rewards.append(output_batch)
        rewards = torch.cat(rewards, dim=0)
        self.rewards = rewards
        #print(rewards)

        # if self.interaction_type == "opt":
        #     test_reward = reward_batch[0]  # 必须是 scalar
        #     test_reward.backward(retain_graph=True)
        #     model = self.calculator
        #     has_grad = False
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"✅ {name} 有梯度！reward 可反向传播")
        #             has_grad = True
        #             break
        #     if not has_grad:
        #         print("❌ 没有梯度，reward 不能反向传播")



    def compute_interactions(self):
        with torch.no_grad():
            self.calculate_all_subset_rewards()

        # self.rewards = get_reward(self.outputs,
        #                           selected_dim=self.selected_dim,
        #                           gt=self.target,
        #                           sample=self.softmax_sample_dims)
        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        if self.verbose:
            print(f"rewards shape: {self.rewards.shape}")

        # we use v(S)-v(empty) to calculate the interactions, it will make I(empty)=0, but other interactions remains the same
        self.I_and = torch.matmul(self.reward2Iand, self.rewards_minus_v0)
        #self.I_or = torch.matmul(self.reward2Ior, self.rewards_minus_v0)


    def compute_interactions_from_rewards_and_masks(self, rewards):  # 这个函数一般用不到
        self.rewards = rewards
        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        self.I_and = torch.matmul(self.reward2Iand, self.rewards_minus_v0)
        #self.I_or = torch.matmul(self.reward2Ior, self.rewards_minus_v0)


    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "rewards.npy"), self.rewards.cpu().numpy())
        np.save(osp.join(save_folder, "rewards_minus_v0.npy"), self.rewards_minus_v0.cpu().numpy())
        np.save(osp.join(save_folder, "player_masks.npy"), self.player_masks.cpu().numpy())
        np.save(osp.join(save_folder, "sample_masks.npy"), self.sample_masks.cpu().numpy())
        np.save(osp.join(save_folder, "I_and.npy"), self.I_and.cpu().numpy())
        np.save(osp.join(save_folder, "sample_probs.npy"), self.p.cpu().numpy())
        #np.save(osp.join(save_folder, "I_or.npy"), self.I_or.cpu().numpy())

    def get_player_masks(self):
        return self.player_masks

    def get_sample_masks(self):
        return self.sample_masks

    def get_and_interaction(self) -> torch.Tensor:
        return self.I_and

    # def get_or_interaction(self) -> torch.Tensor:
    #     return self.I_or

    def get_rewards(self) -> torch.Tensor:
        return self.rewards




class ShapleyTaylor(AndOrHarsanyi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.reward2Ishapley = get_reward2Ishapley_mat(n_dim=self.n_players, sort_type=self.sort_type).to(self.device)
        #self.reward2Iand = get_reward2Iand_mat(n_dim=self.n_players, sort_type=self.sort_type).to(self.device)
    def compute_interactions(self):
        with torch.no_grad():
            self.calculate_all_subset_rewards()
        # self.calculate_all_subset_rewards()

        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        if self.verbose:
            print(f"rewards shape: {self.rewards.shape}")

        # Calculate Shapley Taylor interactions
        k = 2  # Taylor expansion order
        print("rewards_minus_v0 shape:", self.rewards_minus_v0.shape) 
        I_and = torch.matmul(self.reward2Iand, self.rewards_minus_v0)
        #I_and = torch.load()
        # Calculate subset sizes |S|
        subset_sizes = self.player_masks.sum(dim=1)  # [2^n_players]

        # 初始化 I_shapley 为 I_and，然后将 |S| > k 的子集值置零
        self.I_shapley = I_and.clone()
        self.I_shapley[subset_sizes > k] = 0.0  # 新增逻辑：|S| > k 时设为 0

        # 仅处理 |S| = k 的子集
        eq_k_mask = subset_sizes == k
        eq_k_indices = torch.where(eq_k_mask)[0]

        for s_idx in eq_k_indices:
            s_mask = self.player_masks[s_idx]
            # 寻找包含 S 的超集 T
            supersets = (self.player_masks & s_mask).sum(dim=1) == s_mask.sum()
            t_indices = torch.where(supersets)[0]
            print(t_indices)
            # 计算权重: 1/C(|T|,k)
            t_sizes = subset_sizes[t_indices]
            #print(t_sizes)
            weights = 1.0 / torch_comb_safe(t_sizes, k)
            #print(weights)
            # 加权求和并更新 I_shapley
            weighted_sum = (I_and[t_indices] * weights).sum()
            self.I_shapley[s_idx] = weighted_sum

    def get_shapley_interaction(self) -> torch.Tensor:
        return self.I_shapley
    
    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "rewards.npy"), self.rewards.cpu().numpy())
        np.save(osp.join(save_folder, "rewards_minus_v0.npy"), self.rewards_minus_v0.cpu().numpy())
        #np.save(osp.join(save_folder, "player_masks.npy"), self.player_masks.cpu().numpy())
        #np.save(osp.join(save_folder, "sample_masks.npy"), self.sample_masks.cpu().numpy())
        #np.save(osp.join(save_folder, "I_shapley.npy"), self.I_shapley.cpu().numpy())


class ShapleyInteractionIndex(AndOrHarsanyi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward2Ishapley_interaction = get_reward2Ishapley_interaction_mat(n_dim=self.n_players, sort_type=self.sort_type).to(self.device)

    def compute_interactions(self):
        with torch.no_grad():
            self.calculate_all_subset_rewards()

        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        if self.verbose:
            print(f"rewards shape: {self.rewards.shape}")

        # Calculate Shapley Interaction Index
        self.I_shapley_interaction = torch.matmul(self.reward2Ishapley_interaction, self.rewards_minus_v0)

    def get_shapley_interaction_index(self) -> torch.Tensor:
        return self.I_shapley_interaction
    
    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "rewards.npy"), self.rewards.cpu().numpy())
        np.save(osp.join(save_folder, "rewards_minus_v0.npy"), self.rewards_minus_v0.cpu().numpy())

class Shapley(AndOrHarsanyi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.I_shapleyv = None  # 存储Shapley值

    def compute_interactions(self):
        with torch.no_grad():
            self.calculate_all_subset_rewards()

        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        if self.verbose:
            print(f"rewards shape: {self.rewards.shape}")

        # 计算AND交互值
        self.I_and = torch.matmul(self.reward2Iand, self.rewards_minus_v0)

        # 计算Shapley值
        n_players = self.n_players
        self.I_shapleyv = torch.zeros(n_players, device=self.device)

        # 遍历所有玩家
        for i in range(n_players):
            # 找出所有包含玩家i的子集
            contains_i = self.player_masks[:, i]
            subset_indices = torch.where(contains_i)[0]
            
            # 计算每个子集的权重和贡献
            for s_idx in subset_indices:
                s_mask = self.player_masks[s_idx]
                s_size = s_mask.sum().item()
                weight = 1.0 / s_size
                self.I_shapleyv[i] += weight * self.I_and[s_idx]

    def get_shapley_value(self) -> torch.Tensor:
        return self.I_shapleyv
        
    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "I_shapleyv.npy"), self.I_shapleyv.cpu().numpy())
        super().save(save_folder)

class CalculateReward(AndOrHarsanyi):
    def __init__(self, *args, file_path_template,ep_count_file, single_feature_file, type, MP_size, MP_used_nodes, **kwargs):
        super().__init__(*args, **kwargs)
        if type == "attribution":
            communities = []
            # 1. 保留某个 player 的社区（单节点社区）
            for i in range(self.n_players):
                communities.append([i])
            # 2. 去掉某个 player 的社区（除 i 之外的所有节点）
            for i in range(self.n_players):
                communities.append([j for j in range(self.n_players) if j != i])
            self.communities = communities
            print(communities)
        else: self.communities = generate_all_communities(self.n_players, self.sample_id, file_path_template,ep_count_file, single_feature_file, type, MP_size, MP_used_nodes)
        self.player_masks = torch.BoolTensor(generate_all_masks_re(self.n_players, self.sort_type, self.communities)).to(self.device)
        print(self.player_masks)
    def compute_interactions(self):
        # with torch.no_grad():
        #     self.calculate_all_subset_rewards()
        self.calculate_all_subset_rewards()

        self.rewards_minus_v0 = self.rewards - self.rewards[0]
        if self.verbose:
            print(f"rewards shape: {self.rewards.shape}")
    
    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "rewards.npy"), self.rewards.cpu().numpy())
        np.save(osp.join(save_folder, "rewards_minus_v0.npy"), self.rewards_minus_v0.cpu().numpy())
        # Calculate Shapley Interaction Index
        #self.I_shapley_interaction = torch.matmul(self.play_mask, self.rewards_minus_v0)


class ShapleyC(AndOrHarsanyi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.I_shapleyv = None
        self.batch_size = kwargs.get('batch_size', 32)  # 默认批量大小

    def compute_interactions(self):
        from captum.attr import ShapleyValueSampling

        self.forward_function

        input_tensor = self.input.to(self.device)
        baseline = self.baseline

        # 处理基线
        if isinstance(baseline, int):
            baseline = torch.ones_like(input_tensor, device=self.device) * baseline
        elif isinstance(baseline, torch.Tensor):
            baseline = baseline.to(self.device)

        # 创建Shapley值计算器
        shapley_calculator = ShapleyValueSampling(self.forward_function)

        # 分批处理大型输入
        if input_tensor.size(0) > self.batch_size:
            attr_results = []
            for i in range(0, input_tensor.size(0), self.batch_size):
                batch_input = input_tensor[i:i + self.batch_size]
                batch_baseline = baseline[i:i + self.batch_size] if isinstance(baseline, torch.Tensor) else baseline

                attr = shapley_calculator.attribute(
                    inputs=batch_input,
                    baselines=batch_baseline,
                    target=self.target,
                    n_samples=50,
                    show_progress=self.verbose
                )
                attr_results.append(attr)

            attr = torch.cat(attr_results, dim=0)
        else:
            attr = shapley_calculator.attribute(
                inputs=input_tensor,
                baselines=baseline,
                target=self.target,
                n_samples=50,
                show_progress=self.verbose
            )

        self.I_shapleyv = attr.squeeze(0)

        # 自定义玩家子集聚合
        if self.all_players_subset is not None:
            aggregated_shapley = torch.zeros(
                len(self.all_players_subset),
                device=self.device
            )
            for i, player_indices in enumerate(self.all_players_subset):
                aggregated_shapley[i] = self.I_shapleyv[player_indices].sum()
            self.I_shapleyv = aggregated_shapley
            self.rewards = self.I_shapleyv

    def get_shapley_value(self) -> torch.Tensor:
        return self.I_shapleyv

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        # 保存前移回CPU
        np.save(osp.join(save_folder, "I_shapleyv.npy"),
                self.I_shapleyv.cpu().numpy())

class CalculateOpt(AndOrHarsanyi):
    def __init__(self, *args, file_path_template, ep_count_file, single_feature_file, type, **kwargs):
        super().__init__(*args, **kwargs)
        self.communities = generate_all_communities(self.n_players,self.sample_id, file_path_template, ep_count_file, single_feature_file, type, MP_size=0, MP_used_nodes=None)
        self.player_masks = torch.BoolTensor(
            generate_all_masks_re(self.n_players, self.sort_type, self.communities)).to(self.device)
        print(self.player_masks)

    def compute_interactions(self, tau_start=None, tau_end=None, tau_rate=None, k_file=None, opt_path=None, sample_id=None, baseline_value_embeds=None,
                             attention_mask=None):
        self.rewards = torch.zeros(1, device=self.device)
        # 默认 k 是 None，只有当 k_file 给出并且 sample_id 存在时才赋值
        sample_k = None

        if k_file is not None and sample_id is not None:
            k_dict = {}
            with open(k_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':')
                        key = key.strip()
                        value = int(value.strip())
                        k_dict[key] = value
            sample_key = f"sample{sample_id}"
            if sample_key in k_dict:
                sample_k = k_dict[sample_key]

        # 如果 sample_k 是 0 或 None（不在文件中），就跳过
        if sample_k is None:
            print(f"[OPT] sample{sample_id} not in k_file, skipping.")
            return
        if sample_k == 0:
            print(f"[OPT] sample{sample_id} has k=0, skipping.")
            return

        print(f"[OPT] Running optimize_players() with k={sample_k}...")
        # optimize_players(self, sample_k, 300, 1e-3, 30, 1e-3, tau_start, tau_end, tau_rate, opt_path, sample_id, baseline_value_embeds,
        #                  attention_mask)
        summary = optimize_players_mp(
            calculator=self,
            k=sample_k,
            num_steps=1000,
            lr=1e-2,
            patience=5,
            tol=1e-4,
            opt_path=opt_path,
            sample_id=sample_id,
            baseline_value_embeds=baseline_value_embeds,
            attention_mask=attention_mask,
            l1_lambda=1e-3,
            tv_lambda=1e-2,
            tv_beta=1.0,
        )

    def __call__(self, *args, **kwargs):
        return self.calculator(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.calculator, name):
            return getattr(self.calculator, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        np.save(osp.join(save_folder, "rewards.npy"), self.rewards.detach().cpu().numpy())
        # np.save(osp.join(save_folder, "rewards_minus_v0.npy"), self.rewards_minus_v0.cpu().detach().numpy())
        # Calculate Shapley Interaction Index
        # self.I_shapley_interaction = torch.matmul(self.play_mask, self.rewards_minus_v0)

def hard_topk_mask_from_indices(n, topk_indices):
    hard_mask = torch.zeros(n, dtype=torch.float, device=topk_indices.device)
    hard_mask[topk_indices] = 1.0
    return hard_mask.unsqueeze(0)  # [1, n_players]

def get_v_s_hard(calculator, scores, k, baseline_value_embeds, attention_mask):
    from .calculate import get_forward_function_nlp
    from .mask_utils import get_mask_input_function_nlp
    topk_indices = torch.topk(scores, k).indices
    hard_mask = torch.zeros_like(scores)
    hard_mask[topk_indices] = 1.0

    # 保存当前状态
    original_interaction_type = calculator.interaction_type
    original_player_masks = calculator.player_masks
    original_mask_input_fn = calculator.mask_input_function
    original_forward_fn = calculator.forward_function

    # 设置 hard mask 和函数
    calculator.player_masks = (hard_mask.unsqueeze(0) > 0).bool()
    calculator.interaction_type = "hard"
    calculator.mask_input_function = get_mask_input_function_nlp()
    calculator.forward_function = get_forward_function_nlp(
        calculator=calculator,
        baseline_value_embeds=baseline_value_embeds,
        attention_mask=attention_mask
    )

    calculator.calculate_all_subset_rewards()
    v_s_hard = calculator.rewards[0]

    # 恢复原状态（soft）
    calculator.interaction_type = original_interaction_type
    calculator.player_masks = original_player_masks
    calculator.mask_input_function = original_mask_input_fn
    calculator.forward_function = original_forward_fn

    return v_s_hard

def total_variation_1d(x, beta=1.0):
    """
    1D total variation for token-level mask: sum |x[i+1]-x[i]|^beta
    Returns (tv_value, grad proxy) -- but here we only need scalar for loss.
    """
    if x.numel() <= 1:
        return torch.tensor(0.0, device=x.device)
    diffs = x[1:] - x[:-1]
    if beta == 1.0:
        tv = torch.sum(torch.abs(diffs))
    else:
        tv = torch.sum(torch.pow(torch.abs(diffs) + 1e-12, beta))
    return tv

def optimize_players_mp(
    calculator,
    k,
    num_steps,
    lr,
    patience,
    tol,
    tau_start=None,
    tau_end=None,
    tau_rate=None,
    opt_path=".",
    sample_id=0,
    baseline_value_embeds=None,
    attention_mask=None,
    # MP-specific hyperparams
    l1_lambda=1e-3,
    tv_lambda=1e-2,
    tv_beta=1.0,
    mask_init=None,
    mask_param_init_scale=1e-2,
    eval_every=5,
    device=None,
):
    """
    MP-style optimization for token/player masks (Meaningful Perturbation).
    - calculator: object that holds n_players, device, player_masks, calculate_all_subset_rewards(), rewards, ...
    - k: target top-k for final hard evaluation (used by get_v_s_hard)
    - num_steps: maximum steps (None for until patience triggers)
    - lr: learning rate for mask param
    - patience, tol: early stopping on hard v(S) change
    - baseline_value_embeds, attention_mask: forwarded to get_v_s_hard when evaluating hard mask
    - l1_lambda, tv_lambda, tv_beta: MP regularizers
    - mask_init: optional initial mask values (1D numpy or torch size n_players)
    - mask_param_init_scale: std for init noise
    - eval_every: how often (steps) to compute hard v(S)
    Returns: dict with results and paths saved in opt_path.
    """

    # setup
    if device is None:
        device = getattr(calculator, "device", torch.device("cpu"))
    n_players = calculator.n_players
    os.makedirs(os.path.join(opt_path, "log"), exist_ok=True)
    log_file = os.path.join(opt_path, "log", f"sample{sample_id}.txt")
    result_dir = os.path.join(opt_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"sample{sample_id}.txt")

    # initialize mask parameters (sigmoid)
    if mask_init is not None:
        if isinstance(mask_init, np.ndarray):
            init_mask = torch.tensor(mask_init, dtype=torch.float, device=device)
        elif isinstance(mask_init, torch.Tensor):
            init_mask = mask_init.to(device).float()
        else:
            raise ValueError("mask_init must be numpy array or torch tensor")
        eps = 1e-6
        init_mask_clamped = init_mask.clamp(eps, 1.0 - eps)
        mask_param_init = torch.log(init_mask_clamped / (1.0 - init_mask_clamped))
        mask_param = torch.nn.Parameter(mask_param_init.clone().detach())
    else:
        mask_param = torch.nn.Parameter(torch.randn(n_players, device=device) * mask_param_init_scale)

    optimizer = torch.optim.Adam([mask_param], lr=lr)

    # bookkeeping
    loss_curve, v_s_curve = [], []
    prev_v_s_hard = None
    no_improve_count = 0
    best_v_s_hard = -float("inf")
    best_step = -1
    best_v_s_soft_at_peak = None
    best_mask_param = None

    max_steps = num_steps if num_steps is not None else 10000

    # main loop
    with open(log_file, "w") as log_f:
        for step in range(max_steps):
            mask = torch.sigmoid(mask_param)
            mask_batch = mask.unsqueeze(0)

            calculator.player_masks = mask_batch
            calculator.calculate_all_subset_rewards()
            v_s = calculator.rewards[0]
            if not torch.is_tensor(v_s):
                v_s = torch.tensor(v_s, device=device, dtype=torch.float)

            l1_term = (1.0 - mask).sum()
            tv_term = total_variation_1d(mask, beta=tv_beta)
            loss = -v_s + l1_lambda * l1_term + tv_lambda * tv_term

            loss_curve.append(loss.item())
            v_s_curve.append(v_s.item())

            optimizer.zero_grad()
            loss.backward()
            grad_norm = mask_param.grad.norm().item() if mask_param.grad is not None else None
            optimizer.step()

            # logging
            log_f.write(f"[Step {step}] loss={loss.item():.6f}, v_s_soft={v_s.item():.6f}, l1={l1_term.item():.6f}, tv={tv_term.item():.6f}\n")
            log_f.write(f"[Step {step}] grad_norm={grad_norm}\n")
            current_topk = torch.topk(mask, k).indices.tolist()
            log_f.write(f"[Step {step}] soft top-k indices: {current_topk}\n")

            # periodic eval
            if step % eval_every == 0 or step == max_steps - 1:
                try:
                    v_s_hard = get_v_s_hard(
                        calculator=calculator,
                        scores=mask.detach(),
                        k=k,
                        baseline_value_embeds=baseline_value_embeds,
                        attention_mask=attention_mask,
                    )
                    if not torch.is_tensor(v_s_hard):
                        v_s_hard = torch.tensor(v_s_hard, device=device, dtype=torch.float)
                except Exception as e:
                    log_f.write(f"[WARN] get_v_s_hard() failed at step {step}: {e}\n")
                    continue

                log_f.write(f"[Eval Step {step}] v_s_hard = {v_s_hard.item():.6f}\n")

                # Track best
                if v_s_hard.item() > best_v_s_hard:
                    best_v_s_hard = v_s_hard.item()
                    best_step = step
                    best_v_s_soft_at_peak = v_s.item()
                    best_mask_param = mask_param.detach().clone()
                    no_improve_count = 0
                else:
                    if prev_v_s_hard is not None and abs(v_s_hard.item() - prev_v_s_hard) < tol:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0

                prev_v_s_hard = v_s_hard.item()
                log_f.write(f"[Eval Step {step}] best_v_s_hard = {best_v_s_hard:.6f} (step {best_step})\n")
                log_f.flush()

                if num_steps is None and no_improve_count >= patience:
                    print(f"[MP OPT] Early stopping at step {step} (no improvement {no_improve_count}x).")
                    break

    # === Final evaluation ===
    with torch.no_grad():
        final_mask = torch.sigmoid(mask_param)

        # Final (hard) using unified get_v_s_hard
        v_s_hard_final = get_v_s_hard(
            calculator=calculator,
            scores=final_mask.detach(),
            k=k,
            baseline_value_embeds=baseline_value_embeds,
            attention_mask=attention_mask,
        )
        if not torch.is_tensor(v_s_hard_final):
            v_s_hard_final = torch.tensor(v_s_hard_final, device=device, dtype=torch.float)

        # Also record the final soft (for reference)
        calculator.player_masks = final_mask.unsqueeze(0)
        calculator.calculate_all_subset_rewards()
        v_s_soft_final = calculator.rewards[0]
        if not torch.is_tensor(v_s_soft_final):
            v_s_soft_final = torch.tensor(v_s_soft_final, device=device, dtype=torch.float)

        # Peak (best mask param)
        if best_mask_param is not None:
            best_mask = torch.sigmoid(best_mask_param)
            v_s_hard_peak_recomputed = get_v_s_hard(
                calculator=calculator,
                scores=best_mask.detach(),
                k=k,
                baseline_value_embeds=baseline_value_embeds,
                attention_mask=attention_mask,
            )
            if not torch.is_tensor(v_s_hard_peak_recomputed):
                v_s_hard_peak_recomputed = torch.tensor(v_s_hard_peak_recomputed, device=device, dtype=torch.float)
        else:
            v_s_hard_peak_recomputed = torch.tensor(best_v_s_hard, device=device)

    # === Save results ===
    with open(result_file, "w") as f_res:
        f_res.write(f"[Peak] v(S)_hard_max (via get_v_s_hard): {v_s_hard_peak_recomputed.item():.6f}\n")
        f_res.write(f"[Peak] top-k indices: {torch.topk(torch.sigmoid(best_mask_param), k).indices.tolist() if best_mask_param is not None else 'N/A'}\n")
        f_res.write(f"[Peak] step: {best_step}\n")
        f_res.write(f"[Peak] v(S)_soft_at_hard_peak: {best_v_s_soft_at_peak:.6f}\n")
        f_res.write(f"[Final] v(S)_hard_final (via get_v_s_hard): {v_s_hard_final.item():.6f}\n")
        f_res.write(f"[Final] top-k indices: {torch.topk(torch.sigmoid(mask_param), k).indices.tolist()}\n")
        f_res.write(f"[Final] v(S)_soft_final: {v_s_soft_final.item():.6f}\n")

    # # try plotting
    # try:
    #     plt.figure(figsize=(10, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(loss_curve, label="Loss")
    #     plt.xlabel("Step")
    #     plt.ylabel("Loss")
    #     plt.title(f"MP Loss Curve (sample {sample_id})")
    #     plt.legend()
    #
    #     plt.subplot(1, 2, 2)
    #     plt.plot(v_s_curve, label="v(S)_soft")
    #     plt.xlabel("Step")
    #     plt.ylabel("v(S)_soft")
    #     plt.title(f"MP Soft Value Curve (sample {sample_id})")
    #     plt.legend()
    #
    #     plt.tight_layout()
    #     plt_path = os.path.join(opt_path, f"mp_loss_curve_sample{sample_id}.png")
    #     plt.savefig(plt_path)
    #     plt.close()
    # except Exception as e:
    #     print(f"[WARN] Failed to plot MP curves: {e}")

    # return a summary dict
    summary = {
        "final_soft_mask": final_mask.cpu().numpy(),
        "final_hard_topk_indices": torch.topk(torch.sigmoid(mask_param), k).indices.tolist(),
        "v_s_hard_final": float(v_s_hard_final.item()),
        "v_s_soft_final": float(v_s_soft_final.item()),
        "best_v_s_hard": float(best_v_s_hard),
        "best_step": int(best_step),
        "loss_curve": loss_curve,
        "v_s_curve": v_s_curve,
        "result_file": result_file,
        "log_file": log_file,
        "plot_path": plt_path if 'plt_path' in locals() else None,
    }

    return summary

# def soft_topk(scores, k, temperature=0.01):  # 0.01 比 1.0 更尖锐
#     scores_sorted, _ = torch.sort(scores, descending=True)
#     threshold = scores_sorted[k-1]
#     mask = torch.sigmoid((scores - threshold) / temperature)
#     mask = (mask / mask.sum()) * k
#     return mask

# def gumbel_soft_topk(logits, k, tau, hard=True):
#     # Sample Gumbel noise
#     noise = torch.empty_like(logits).uniform_(0, 1)
#     gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
#
#     # Add noise and divide by tau
#     y = (logits + gumbel_noise) / tau
#
#     # Softmax over logits, then rescale so sum ≈ k
#     y_soft = F.softmax(y, dim=-1) * k
#
#     if hard:
#         # Hard top-k over noisy logits (NOT softmax!)
#         topk = torch.topk(y, k)
#         hard_mask = torch.zeros_like(logits)
#         hard_mask.scatter_(0, topk.indices, 1.0)
#         return (hard_mask - y_soft).detach() + y_soft  # STE
#     else:
#         return y_soft

# def optimize_players(calculator, k, num_steps, lr, patience, tol, tau_start, tau_end, tau_rate,
#                      opt_path=None, sample_id=None, baseline_value_embeds=None, attention_mask=None):
#
#     n_players = calculator.n_players
#     scores = torch.nn.Parameter(
#         torch.zeros(n_players, device=calculator.device) + torch.randn(n_players, device=calculator.device) * 1e-3
#     )
#     optimizer = torch.optim.Adam([scores], lr=lr)
#
#     os.makedirs(os.path.join(opt_path, "log"), exist_ok=True)
#     log_file = os.path.join(opt_path, "log", f"sample{sample_id}.txt")
#     topk_file = os.path.join(opt_path, f"sample{sample_id}.txt")
#
#     loss_curve = []
#     v_s_curve = []
#     prev_v_s_hard = None
#     no_improve_count = 0
#     best_v_s_hard = float('-inf')
#     max_steps = num_steps if num_steps is not None else 10_000
#
#     with open(log_file, "w") as log_f:
#         for step in range(max_steps):
#             tau = max(tau_end, tau_start * (tau_rate ** step))
#             # if step % 10 == 0:
#             #     mask = gumbel_soft_topk(scores, k, tau=tau, hard=True).unsqueeze(0)
#             # else:
#             #     soft_mask = F.softmax(scores / tau, dim=-1) * k
#             #     mask = (hard_topk_mask_from_indices(n_players,
#             #                                         torch.topk(scores, k).indices) - soft_mask).detach() + soft_mask
#             mask = gumbel_soft_topk(scores, k, tau=tau, hard=True).unsqueeze(0)
#             calculator.player_masks = mask
#             calculator.calculate_all_subset_rewards()
#             v_s = calculator.rewards[0]
#             v_s_curve.append(v_s.item())
#
#             v_s_hard = get_v_s_hard(calculator, scores, k, baseline_value_embeds, attention_mask)
#
#             loss = -(v_s_hard + v_s - v_s.detach())
#             loss_curve.append(loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # if step % 5 == 0:
#             log_f.write(f"[Step {step}] loss = {loss.item():.4f}, v(S)_soft = {v_s.item():.4f}, v(S)_hard = {v_s_hard.item():.4f}\n")
#             grad_norm = scores.grad.norm().item() if scores.grad is not None else 'None'
#             log_f.write(f"[Step {step}] grad norm: {grad_norm}\n")
#             current_topk = torch.topk(scores, k).indices.tolist()
#             log_f.write(f"[Step {step}] top-k indices: {current_topk}\n")
#             with open(topk_file, "a") as f_append:
#                 f_append.write(f"[Step {step}] top-k indices: {current_topk}\n")
#
#             if prev_v_s_hard is not None and abs(v_s_hard.item() - prev_v_s_hard) < tol:
#                 no_improve_count += 1
#             else:
#                 no_improve_count = 0
#
#             if v_s_hard.item() > best_v_s_hard:
#                 best_v_s_hard = v_s_hard.item()
#                 best_step = step
#                 best_scores = scores.detach().clone()
#
#             prev_v_s_hard = v_s_hard.item()
#
#             if num_steps is None and no_improve_count >= patience:
#                 print(f"[OPT] Early stopping at step {step} with v(S) = {v_s.item():.4f}")
#                 break
#
#         # Get hard Top-K indices from final scores (no noise)
#         topk_indices = torch.topk(scores, k).indices
#
#         # Final hard mask for evaluation
#         final_hard_mask = hard_topk_mask_from_indices(n_players, topk_indices)
#         calculator.player_masks = final_hard_mask
#
#         # Soft mask from plain softmax over raw scores (no Gumbel)
#         soft_mask_final = F.softmax(scores / 1.0, dim=-1) * k
#
#         # For debug: soft values at hard mask positions
#         soft_values_at_hard = soft_mask_final[topk_indices]
#
#         log_f.write(f"[DEBUG] Final raw soft mask (no Gumbel):\n")
#         log_f.write(f"{soft_mask_final.tolist()}\n")
#         log_f.write(f"[DEBUG] Hard mask indices (top-k of scores): {topk_indices.tolist()}\n")
#         log_f.write(f"[DEBUG] Soft mask values at hard mask indices:\n")
#         log_f.write(f"{soft_values_at_hard.tolist()}\n")
#
#         # Also compute top-k indices from soft mask for comparison
#         soft_topk_indices = torch.topk(soft_mask_final, k).indices.tolist()
#         log_f.write(f"[DEBUG] Soft mask top-k indices (for comparison): {soft_topk_indices}\n")
#
#         # Final evaluation
#         v_s_hard_final = get_v_s_hard(calculator, scores, k, baseline_value_embeds, attention_mask).item()
#         log_f.write(f"[HARD MASK] v(S): {v_s_hard_final:.4f}\n")
#         log_f.write(f"[COMPARE] Last soft v(S): {v_s.item():.4f}, Final hard v(S): {v_s_hard_final:.4f}\n")
#         log_f.write(f"[COMPARE] Delta = {abs(v_s.item() - v_s_hard_final):.4f}\n")
#
#
#     with open(topk_file, "a") as f:
#         f.write(f"[Final] top-k indices: {topk_indices.tolist()}\n")
#
#     try:
#         plt.figure(figsize=(10, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(loss_curve, label="Loss")
#         plt.xlabel("Step")
#         plt.ylabel("Loss")
#         plt.title(f"Loss Curve (sample {sample_id})")
#         plt.legend()
#
#         plt.subplot(1, 2, 2)
#         plt.plot(v_s_curve, label="v(S)_soft", color='green')
#         plt.xlabel("Step")
#         plt.ylabel("v(S)_soft")
#         plt.title(f"Soft Value Curve (sample {sample_id})")
#         plt.legend()
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(opt_path, f"loss_curve_sample{sample_id}.png"))
#         plt.close()
#
#     except Exception as e:
#         print(f"[WARN] Failed to plot loss curve: {e}")
#
#     # === Save best v_s_hard and corresponding top-k indices ===
#     best_topk_indices = torch.topk(best_scores, k).indices
#     result_file = os.path.join(opt_path, f"result/sample{sample_id}.txt")
#     os.makedirs(os.path.dirname(result_file), exist_ok=True)
#
#     with open(result_file, "w") as f_result:
#         f_result.write(f"[Best] top-k indices: {best_topk_indices}\n")
#         f_result.write(f"[Best] v(S)_hard: {best_v_s_hard:.4f}\n")
#         f_result.write(f"[Best] reached at step: {best_step}\n")

