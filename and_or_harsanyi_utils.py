import torch
import math
from .set_utils import generate_all_masks, generate_subset_masks, generate_reverse_subset_masks, \
    generate_set_with_intersection_masks
from tqdm import tqdm
from math import comb

# def get_reward2Iand_mat(n_dim: int, sort_type: str):
#     '''
#     The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
#     :param n_dim: the input dimension n
#     :param sort_type: the type of sorting the masks
#     :return a matrix, with shape 2^n * 2^n
#     '''
#     all_masks = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type))
#     n_masks, _ = all_masks.shape
#     mat = []
#     for i in tqdm(range(n_masks), ncols=100, desc="Generating reward2Iand matrix"):
#     # for i in range(n_masks):
#         mask_S = all_masks[i]
#         s_sum = mask_S.sum().item()
#         row = torch.zeros(n_masks)
#         # if s_sum > 2:
#         #     # 如果S的大小超过2，直接添加全零行
#         #     mat.append(row)
#         #     continue
#         # ===============================================================================================
#         # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
#         # mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks) # baseline = ()
#         mask_Ls, L_indices = generate_reverse_subset_masks(mask_S, all_masks) # baseline = N
#         L_indices = (L_indices == True).nonzero(as_tuple=False)
#         assert mask_Ls.shape[0] == L_indices.shape[0]
#         row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
#         # ===============================================================================================
#         mat.append(row.clone())
#     mat = torch.stack(mat).float() # todo: 这个float可能要改一下
#     return mat
def get_reward2Iand_mat(n_dim: int, sort_type: str, reverse: bool = False):
    '''
    The transformation matrix from reward to interaction.
    Supports both standard Harsanyi interaction (reverse=False) 
    and removal-based interaction (reverse=True).
    '''
    # 1. 生成所有掩码
    # 注意：这里生成的 masks 列表顺序对应矩阵的行（计算的目标交互）和列（已知的效用值）
    all_masks_list = generate_all_masks(n_dim, sort_type=sort_type, reverse=reverse) # 这里k硬编码为2以保持一致
    all_masks = torch.BoolTensor(all_masks_list)
    n_masks, _ = all_masks.shape
    
    mat = []
    for i in tqdm(range(n_masks), ncols=100, desc="Generating reward2Iand matrix"):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        
        if not reverse:
            # ==============================================================================
            # Case 1: Standard Harsanyi Interaction (reverse=False)
            # Formula: I(S) = \sum_{L \subseteq S} (-1)^{|S| - |L|} v(L)
            # mask_S represents S (items present)
            # We look for mask_L representing L such that L is a subset of S
            # ==============================================================================
            
            # 找到所有是 mask_S 子集的 mask_L
            # L \subseteq S  <==>  (S & L) == L
            is_subset = (mask_S.unsqueeze(0) & all_masks) == all_masks
            # is_subset shape: [n_masks, n_dim] -> all check -> [n_masks]
            is_subset = is_subset.all(dim=1)
            
            L_indices = is_subset.nonzero(as_tuple=False).squeeze()
            mask_Ls = all_masks[L_indices]
            
            if L_indices.dim() == 0: # handle single match case
                L_indices = L_indices.unsqueeze(0)
                mask_Ls = mask_Ls.unsqueeze(0)

            # Calculate (-1)^{|S| - |L|}
            # |S| = mask_S.sum()
            # |L| = mask_Ls.sum(dim=1)
            coeffs = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1))
            row[L_indices] = coeffs
            
        else:
            # ==============================================================================
            # Case 2: Removal Interaction (reverse=True)
            # Formula: I(S_removed) = \sum_{L_removed \subseteq S_removed} (-1)^{|L_removed|} v(N \setminus L_removed)
            # mask_S represents N \setminus S_removed (items kept)
            # mask_L represents N \setminus L_removed (items kept)
            # Condition: L_removed \subseteq S_removed 
            #         <==> N \setminus S_removed \subseteq N \setminus L_removed
            #         <==> mask_S \subseteq mask_L
            # ==============================================================================
            
            # 找到所有是 mask_S 超集的 mask_L (即 mask_S 是 mask_L 的子集)
            # mask_S \subseteq mask_L <==> (mask_S & mask_L) == mask_S
            is_superset = (mask_S.unsqueeze(0) & all_masks) == mask_S.unsqueeze(0)
            is_superset = is_superset.all(dim=1)
            
            L_indices = is_superset.nonzero(as_tuple=False).squeeze()
            mask_Ls = all_masks[L_indices]
            
            if L_indices.dim() == 0:
                L_indices = L_indices.unsqueeze(0)
                mask_Ls = mask_Ls.unsqueeze(0)

            # Calculate (-1)^{|L_removed|}
            # |L_removed| = N - |N \setminus L_removed| = n_dim - mask_Ls.sum(dim=1)
            coeffs = torch.pow(-1., n_dim - mask_Ls.sum(dim=1).float())
            row[L_indices] = coeffs

        mat.append(row)
        
    mat = torch.stack(mat).float()
    return mat

def get_reward2Ior_mat(n_dim: int, sort_type: str):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to or-interaction
    :param n_dim: the input dimension n
    :param sort_type: the type of sorting the masks
    :return a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type,k=2))
    n_masks, _ = all_masks.shape
    mat = []
    for i in tqdm(range(n_masks), ncols=100, desc="Generating reward2Ior matrix"):
    # for i in range(n_masks):
        mask_S = all_masks[i]
        s_sum = mask_S.sum().item()
        row = torch.zeros(n_masks)
        # if s_sum > 2:
        #     # 如果S的大小超过2，直接添加全零行
        #     mat.append(row)
        #     continue
        # ===============================================================================================
        # Note: I(S) = -\sum_{L\subseteq S} (-1)^{s+(n-l)-n} v(N\L) if S is not empty
        if mask_S.sum() == 0:
            row[i] = 1.
        else:
            mask_NLs, NL_indices = generate_reverse_subset_masks(mask_S, all_masks)
            NL_indices = (NL_indices == True).nonzero(as_tuple=False)
            assert mask_NLs.shape[0] == NL_indices.shape[0]
            row[NL_indices] = - torch.pow(-1., mask_S.sum() + mask_NLs.sum(dim=1) + n_dim).unsqueeze(1)
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    print(mat.shape())
    return mat


def get_reward2Ishapley_mat(n_dim: int, sort_type: str, k: int = 2):
    '''
    The transformation matrix from reward to Taylor interaction index
    :param n_dim: input dimension n
    :param sort_type: sorting type of masks
    :param k: cutoff dimension for Taylor expansion
    :return: matrix of shape C(n,<=k) * C(n,<=k)
    '''
    # 生成两种mask集合
    all_masks_k = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type, k=2))  # 原始k限制的mask集合
    all_masks_full = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type))    # 完整mask集合
    
    # 预计算Harsanyi交互矩阵
    harsanyi_mat = get_reward2Iand_mat(n_dim, sort_type)  # 使用完整mask集合
    
    # 创建索引映射
    mask_to_index_k = {tuple(mask.tolist()): idx for idx, mask in enumerate(all_masks_k)}
    mask_to_index_full = {tuple(mask.tolist()): idx for idx, mask in enumerate(all_masks_full)}
    
    n_masks, _ = all_masks_k.shape
    mat = []
    
    for i in tqdm(range(n_masks), ncols=100, desc="Generating Taylor interaction matrix"):
        mask_S = all_masks_k[i]
        s_size = mask_S.sum().item()
        row = torch.zeros(n_masks)
        
        if s_size < k:
            # 直接获取Harsanyi值
            full_idx = mask_to_index_full[tuple(mask_S.tolist())]
            row[i] = harsanyi_mat[full_idx, full_idx]
        elif s_size == k:
            # 泰勒展开公式: I^Taylor(S) = Σ_{S⊆T} I(T) * (1/C_t^k), 其中t=|T|
            mask_Ts_full, T_indices_full = generate_subset_masks(mask_S, all_masks_full)
            T_indices_full = (T_indices_full == True).nonzero(as_tuple=False)
            
            # 计算泰勒交互值
            taylor_value = 0.0
            for t_idx in T_indices_full:
                mask_T = all_masks_full[t_idx]
                t_size = mask_T.sum().item()
                if t_size >= 2:
                    # 获取Harsanyi交互行并计算加权行
                    harsanyi_row = harsanyi_mat[t_idx]
                    weight = 1.0 / comb(t_size, 2)
                    weighted_row = harsanyi_row * weight
                    # 确保taylor_value是标量
                    taylor_value = weighted_row.sum()
            
            # 将结果赋给对应的k-mask位置
            mask_S_np = mask_S.cpu().numpy().astype(bool).flatten()
            mask_S_tuple = tuple(mask_S_np.tolist())
            if mask_S_tuple in mask_to_index_k:
                s_idx_k = mask_to_index_k[mask_S_tuple]
                if not isinstance(taylor_value, torch.Tensor):
                    taylor_value = torch.tensor(taylor_value, device=row.device)
                # 确保赋值形状匹配
                if taylor_value.dim() > 0:
                    taylor_value = taylor_value.squeeze()
                row[s_idx_k] = taylor_value
            
        mat.append(row.clone())
    
    return torch.stack(mat).float()


def get_reward2Ishapley_interaction_mat(n_dim: int, sort_type: str):
    '''
    The transformation matrix from reward to Shapley Interaction Index
    I_S^v(S) = \sum_{T⊆N\S} \frac{(n - s - t)! t!}{(n - s + 1)!} \sum_{L⊆S} (-1)^{s-l} v(L∪T)
    :param n_dim: the input dimension n
    :param sort_type: the type of sorting the masks
    :return a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type,k=2))
    n_masks, _ = all_masks.shape
    mask_to_index = {tuple(mask.tolist()): idx for idx, mask in enumerate(all_masks)}
    
    mat = []
    for i in tqdm(range(n_masks), ncols=100, desc="Generating Shapley Interaction matrix"):
        mask_S = all_masks[i]
        s_size = mask_S.sum().item()
        mask_NminusS = ~mask_S
        if s_size > 2:
            # 如果S的大小超过2，直接添加全零行
            mat.append(row)
            continue
        # Generate all T ⊆ N\S
        mask_Ts, _ = generate_subset_masks(mask_NminusS, all_masks)
        
        # Generate all L ⊆ S
        mask_Ls, _ = generate_subset_masks(mask_S, all_masks)
        
        row = torch.zeros(n_masks)
        for mask_T in mask_Ts:
            t_size = mask_T.sum().item()
            n_minus_s = n_dim - s_size
            
            # Calculate coefficient: (n-s-t)! t! / (n-s+1)!
            coeff = (math.factorial(n_minus_s - t_size) * math.factorial(t_size)) / math.factorial(n_minus_s + 1)
            
            for mask_L in mask_Ls:
                l_size = mask_L.sum().item()
                sign = (-1) ** (s_size - l_size)
                mask_K = (mask_L | mask_T).bool()
                k_idx = mask_to_index[tuple(mask_K.tolist())]
                row[k_idx] += coeff * sign
        
        mat.append(row)
    
    return torch.stack(mat).float()


def get_Iand2reward_mat(n_dim: int, sort_type: str):
    all_masks = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type,k=2))
    n_masks, _ = all_masks.shape
    mat = []
    for i in tqdm(range(n_masks), ncols=100, desc="Generating Iand2reward matrix"):
    # for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Ior2reward_mat(n_dim: int, sort_type: str):
    all_masks = torch.BoolTensor(generate_all_masks(n_dim, sort_type=sort_type,k=2))
    n_masks, _ = all_masks.shape
    mat = []
    mask_empty = torch.zeros(n_dim).bool()
    _, empty_indice = generate_subset_masks(mask_empty, all_masks)
    for i in tqdm(range(n_masks), ncols=100, desc="Generating Ior2reward matrix"):
    # for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = I(\emptyset) + \sum_{L: L\union S\neq \emptyset} I(S)
        row[empty_indice] = 1.
        mask_Ls, L_indices = generate_set_with_intersection_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat
