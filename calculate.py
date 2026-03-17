from typing import Callable, Union, List, Dict
import torch
import numpy as np
import os
import os.path as osp
from models.nlp import Calculator
from .and_or_harsanyi import AndOrHarsanyi, ShapleyTaylor, ShapleyInteractionIndex, CalculateReward, Shapley, \
    CalculateOpt, ShapleyC
from .set_utils import flatten
from ..player import get_player_words_from_ids
from .mask_utils import get_mask_input_function_nlp, get_mask_input_function_nlp_opt
from .baseline_value import get_baseline_id_nlp
from utils.global_const import *
from utils import LogWriter
import time


gReverse = False
def get_forward_function_nlp(calculator: Calculator,
                             baseline_value_embeds: torch.Tensor,
                             attention_mask: torch.LongTensor = None) -> Callable:
    """
    Get the forward function for the model
    :param model: the Calculator wrapper of the model
    :param baseline_value_embeds: the embedding vector of the baseline value for the model
        [Important note] Different from other models (image, tabular, etc.), in the mask_input_function,
            we first indicate the tokens to be masked with a baseline_flag (an int) for an NLP model.
        Then in this forward function, we replace the embeddings of these flagged tokens with the
            baseline value embedding. The baseline value embeddings can either be learned embeddings or simply
            the embedding of a specific baseline token (e.g., the <unk> or <pad> token)
    :param attention_mask: the attention mask for the input_ids
    :return: the forward function
    """
    def forward_function(input_ids):
        with torch.no_grad():
            mask = (input_ids == BASELINE_FLAG_NLP)  # we use a baseline flag (set to a specific int) to identify the masked positions
            input_ids[mask] = 0  # temporarily set the baseline flags (which is not a valid input id) to 0, in order to run the get_embeds function
            inputs_embeds = calculator.get_embeds(input_ids) # shape (batch_size, seq_len, embed_dim)
            inputs_embeds[mask] = baseline_value_embeds # broadcast the baseline value (embed_dim,) -> (batch_size, seq_len, embed_dim)
            attention_mask_expand = attention_mask.expand_as(input_ids).clone() # [Important] expand the attention mask to the same shape as input_ids (batch_size, seq_len)
            scores = calculator(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask_expand)
        return scores
    return forward_function

def get_forward_function_nlp_opt(calculator: Calculator,
                                 attention_mask: torch.LongTensor = None) -> Callable:
    def forward_function(*, inputs_embeds: torch.Tensor, attention_mask: torch.LongTensor = None):
        """
        inputs_embeds: FloatTensor, shape (batch, seq_len, emb_dim)
        attention_mask: LongTensor, shape (batch, seq_len)
        """
        scores = calculator(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # print(f"[forward_fn_opt] inputs_embeds.norm(): {inputs_embeds.norm().item():.4f}")
        # print(f"[forward_fn_opt] attention_mask.sum(): {attention_mask.sum().item() if attention_mask is not None else 'None'}")

        return scores

    return forward_function



def get_pred_label_nlp(calculator: Calculator,
                       input_ids: torch.LongTensor,
                       attention_mask: torch.LongTensor = None):
    # 在不计算梯度的情况下，计算输入的得分
    # with torch.no_grad():
    #     scores = calculator(input_ids=input_ids,
    #                         attention_mask=attention_mask)
    scores = calculator(input_ids=input_ids,
                        attention_mask=attention_mask)
    print(scores.shape)
    label = torch.argmax(scores, dim=-1).squeeze().item()
    return label, scores


def get_softmax_sample_dims_nlp(calculator: Calculator,
                                input_ids: torch.LongTensor,
                                attention_mask: torch.LongTensor = None,
                                dimension_num: int = 1000):
    """
    Sample a number of dimensions in the logits
    """
    # with torch.no_grad():
    #     scores = calculator(input_ids=input_ids,
    #                         attention_mask=attention_mask)
    #     scores_ = scores.squeeze().clone()
    #     order = torch.argsort(-scores_)
    #     sample_intervals = np.round(np.linspace(0, scores_.shape[0] - 1, dimension_num))
    #     softmax_sample_dims = order[sample_intervals]
    scores = calculator(input_ids=input_ids,
                        attention_mask=attention_mask)
    scores_ = scores.squeeze().clone()
    order = torch.argsort(-scores_)
    sample_intervals = np.round(np.linspace(0, scores_.shape[0] - 1, dimension_num))
    softmax_sample_dims = order[sample_intervals]
    return softmax_sample_dims

def _is_whitespace_only_description(description: str) -> bool:
    """Ignore descriptions that contain only the Ġ token (possibly repeated)."""
    if not description:
        return False
    return all(ch == 'Ġ' for ch in description)

def _prepare_player_descriptions(player_descriptions):
    """Ensure we keep track of which player descriptions should be ignored."""
    if player_descriptions is None:
        return None, set()
    player_descriptions = np.asarray(player_descriptions)
    ignored_player_indices = {i for i, d in enumerate(player_descriptions)
                              if _is_whitespace_only_description(d)}
    return player_descriptions, ignored_player_indices

def log_interaction(interaction_type,save_path, player_ids, player_masks, I_and, player_descriptions=None):
    if isinstance(I_and, torch.Tensor):
        if interaction_type == "opt":
            I_and = I_and.detach().cpu().numpy()
        else:
            #interaction_sum=torch.sum(I_and)
            I_and = I_and.cpu().numpy()
    # if isinstance(I_or, torch.Tensor):
    #     I_or = I_or.cpu().numpy()
    if isinstance(player_masks, torch.Tensor):
        if interaction_type == "opt":
            player_masks = player_masks.detach().cpu().numpy()
        else:
            player_masks = player_masks.cpu().numpy()

    log_interaction = LogWriter(osp.join(save_path, "interaction.txt"), verbose=False, write_mode='w')
    log_interaction_sum = LogWriter(osp.join(save_path, "interaction_sum.txt"), verbose=False, write_mode='w')
    log_nums = 65538  # number of interactions to log todo: 这个之后改一下，可以作为参数传进来，或者更灵活一些

    player_descriptions, ignored_player_indices = _prepare_player_descriptions(player_descriptions)

    if player_descriptions is not None:
        log_interaction.cprint("-"*10 + " Players " + "-"*10)
        for i, d in enumerate(player_descriptions):
            if i in ignored_player_indices:
                continue
            log_interaction.cprint(f"Player {i}: {d}")

    player_names = np.array([i for i in range(len(player_ids))])
    # print([repr(name) for name in player_descriptions])
    # for i, name in enumerate(player_descriptions): print(i, repr(name), I_and[i].item()) 
    log_interaction.cprint("-"*10 + " AND Interactions (Pairwise Only) " + "-"*10)
    log_interaction_sum.cprint("-"*10 + " AND Interactions " + "-"*10)
    interaction_sum = 0.0
    and_order = np.argsort(-np.abs(I_and))[:log_nums]
    for i in and_order:
        #print(player_names[~player_masks[i]].tolist(), player_names[player_masks[i]].tolist())
        if gReverse == True:
            # print("Reverse")
            # print(player_names[~player_masks[i]].tolist())
            coalition = player_names[~player_masks[i]].tolist()
        else:
            coalition = player_names[player_masks[i]].tolist()
        if ignored_player_indices and any(j in ignored_player_indices for j in coalition):
            continue
        # if len(coalition) == 2:  # Only show pairwise interactions
        interaction = I_and[i]
        #if np.abs(interaction) >= np.abs(I_and[and_order[0]]) * 0.025:  # try 10% -> 5% -> 2.5%
        if player_descriptions is None:
            log_str = f'I({",".join(map(str, coalition))}): {interaction}'
        else:
            if gReverse == True:
                coalition_descriptions = player_descriptions[~player_masks[i]].tolist()
            else:
                coalition_descriptions = player_descriptions[player_masks[i]].tolist()
            log_str = f'I({",".join(map(str, coalition))}): {interaction}\t ([{"][".join(coalition_descriptions)}])'
        log_interaction.cprint(log_str)
        # else :
        #     break
        #if len(coalition) == 1 or len(coalition) == 2:
        interaction1 = I_and[i]
        interaction_sum += interaction1
        if player_descriptions is None:
            log_str = f'I({",".join(map(str, coalition))}): {interaction1}'
        else:
            if gReverse == True:
                coalition_descriptions = player_descriptions[~player_masks[i]].tolist()
            else:
                coalition_descriptions = player_descriptions[player_masks[i]].tolist()
            log_str = f'I({",".join(map(str, coalition))}): {interaction1}\t ([{"][".join(coalition_descriptions)}])'
        log_interaction_sum.cprint(log_str)
    log_interaction.cprint(f"Sum: {interaction_sum}")
    log_interaction_sum.cprint(f"Sum: {interaction_sum}")
    # if player_descriptions is not None:
    #     if isinstance(player_descriptions, List):
    #         player_descriptions = np.array(player_descriptions)

    #     log_interaction.cprint("-"*10 + " Players " + "-"*10)
    #     for i, d in enumerate(player_descriptions):
    #         # log_interaction.cprint(f"Player {chr(i + ord('A'))}: {d}")
    #         log_interaction.cprint(f"Player {i}: {d}")

    # # player_names = np.array([chr(i + ord('A')) for i in range(len(player_ids))])
    # player_names = np.array([str(i) for i in range(len(player_ids))])

    # log_interaction.cprint("-"*10 + " AND Interactions (Pairwise Only) " + "-"*10)
    # log_interaction_sum.cprint("-"*10 + " AND Interactions " + "-"*10)
    # interaction_sum = 0.0
    # and_order = np.argsort(-np.abs(I_and))[:log_nums]
    # for i in and_order:
    #     if interaction_type == "opt":
    #         coalition = player_names[player_masks[i].astype(bool)].tolist()
    #     else:
    #         coalition = player_names[player_masks[i]].tolist()
    #     # if len(coalition) == 2:  # Only show pairwise interactions
    #     if 1:
    #         interaction = I_and[i]
    #         if np.abs(interaction) >= np.abs(I_and[and_order[0]]) * 0.025:  # try 10% -> 5% -> 2.5%
    #             if player_descriptions is None:
    #                 log_str = f'I({" ".join(coalition)}): {interaction}'
    #             else:
    #                 if interaction_type == "opt":
    #                     coalition_descriptions = player_descriptions[player_masks[i].astype(bool)].tolist()
    #                 else:
    #                     coalition_descriptions = player_descriptions[player_masks[i]].tolist()
    #                 log_str = f'I({" ".join(coalition)}): {interaction}\t ([{"][".join(coalition_descriptions)}])'
    #             log_interaction.cprint(log_str)
    #         else :
    #             break
    #     if len(coalition) == 1 or len(coalition) == 2:
    #         interaction1 = I_and[i]
    #         interaction_sum += interaction1
    #         if player_descriptions is None:
    #             log_str = f'I({" ".join(coalition)}): {interaction1}'
    #         else:
    #             if interaction_type == "opt":
    #                 coalition_descriptions = player_descriptions[player_masks[i].astype(bool)].tolist()
    #             else:
    #                 coalition_descriptions = player_descriptions[player_masks[i]].tolist()
    #             log_str = f'I({" ".join(coalition)}): {interaction1}\t ([{"][".join(coalition_descriptions)}])'
    #         log_interaction_sum.cprint(log_str)
    # log_interaction.cprint(f"Sum: {interaction_sum}")
    # log_interaction_sum.cprint(f"Sum: {interaction_sum}")

    # log_interaction.cprint("-"*10 + " OR Interactions (Pairwise Only) " + "-"*10)
    # or_order = np.argsort(-np.abs(I_or))[:log_nums]
    # for i in or_order:
    #     coalition = player_names[player_masks[i]].tolist()
    #     if len(coalition) == 2:  # Only show pairwise interactions
    #         interaction = I_or[i]
    #         if player_descriptions is None:
    #             log_str = f'I({"".join(coalition)}): {interaction}'
    #         else:
    #             coalition_descriptions = player_descriptions[player_masks[i]].tolist()
    #             log_str = f'I({"".join(coalition)}): {interaction}\t ([{"][".join(coalition_descriptions)}])'
    #         log_interaction.cprint(log_str)
    log_interaction.cprint("="*50) # separator
    log_interaction.close()
    log_interaction_sum.cprint("="*50) # separator
    log_interaction_sum.close()

def log_rewards(interaction_type,save_path, player_ids, player_masks, I_rewards, player_descriptions=None):
    if isinstance(I_rewards, torch.Tensor):
        if interaction_type == "opt":
            I_rewards = I_rewards.detach().cpu().numpy()
        else:
            I_rewards = I_rewards.cpu().numpy()
    if isinstance(player_masks, torch.Tensor):
        if interaction_type == "opt":
            player_masks = player_masks.detach().cpu().numpy()
        else:
            player_masks = player_masks.cpu().numpy()


    log_rewards = LogWriter(osp.join(save_path, "rewards.txt"), verbose=False, write_mode='w')
    log_nums = 65538  # number of rewards to log todo: 这个之后改一下，可以作为参数传进来，或者更灵活一些
    player_descriptions, ignored_player_indices = _prepare_player_descriptions(player_descriptions)

    if player_descriptions is not None:
        log_rewards.cprint("-"*10 + " Players " + "-"*10)
        for i, d in enumerate(player_descriptions):
            if i in ignored_player_indices:
                continue
            log_rewards.cprint(f"Player {i}: {d}")

    player_names = np.array([i for i in range(len(player_ids))])
    I_rewards_v0 = []

    log_rewards.cprint("-"*10 + " Rewards " + "-"*10)
    print(I_rewards)
    for i in range(len(I_rewards)):
        reward = I_rewards[i]
        if gReverse:
            coalition = player_names[~player_masks[i]].tolist()
            I_rewards_v0.append(reward-I_rewards[0])
        else:
            coalition = player_names[player_masks[i]].tolist()
            I_rewards_v0.append(reward-I_rewards[0])
        if ignored_player_indices and any(j in ignored_player_indices for j in coalition):
            continue
        if player_descriptions is None:
            log_str = f'v({",".join(map(str, coalition))}): {reward}'
        else:
            coalition_descriptions = player_descriptions[player_masks[i]].tolist()
            log_str = f'v({",".join(map(str, coalition))}): {reward}\t ([{"][".join(coalition_descriptions)}])'
        log_rewards.cprint(log_str)
    log_rewards.cprint("-"*10 + " Rewards-v() " + "-"*10)
    I_rewards_v0 = np.array(I_rewards_v0) 
    #and_order = np.argsort(-I_rewards_v0)[:log_nums] #交互图方法排序
    and_order = np.arange(len(I_rewards_v0)) #单归因不排序
    for i in and_order:
        if gReverse == True:
            coalition = player_names[~player_masks[i]].tolist()
        else:
            coalition = player_names[player_masks[i]].tolist()
        if ignored_player_indices and any(j in ignored_player_indices for j in coalition):
            continue
        reward_v0 = I_rewards[i]-I_rewards[0]
        #print(f"reward:{I_rewards[i]},reward_v0:{reward_v0}")
        if player_descriptions is None:
            log_str1 = f'v({",".join(map(str, coalition))}): {reward_v0}'
        else:
            if gReverse == True:
                coalition_descriptions = player_descriptions[player_masks[i]].tolist()
                log_str1 = f'v({",".join(map(str, coalition))}): {reward_v0}, EP: {reward_v0/I_rewards_v0[-1]}\t ([{"][".join(coalition_descriptions)}])'
            else:
                coalition_descriptions = player_descriptions[player_masks[i]].tolist()
            # coalition_descriptions = player_descriptions[player_masks[i]].tolist()
                log_str1 = f'v({",".join(map(str, coalition))}): {reward_v0}, EP: {reward_v0/I_rewards_v0[-1]}\t ([{"][".join(coalition_descriptions)}])'
        log_rewards.cprint(log_str1)
    log_rewards.cprint("="*50) # separator
    log_rewards.close()

    # if player_descriptions is not None:
    #     if isinstance(player_descriptions, List):
    #         player_descriptions = np.array(player_descriptions)

    #     log_rewards.cprint("-"*10 + " Players " + "-"*10)
    #     for i, d in enumerate(player_descriptions):
    #         # log_rewards.cprint(f"Player {chr(i + ord('A'))}: {d}")
    #         log_rewards.cprint(f"Player {i}: {d}")

    # # player_names = np.array([chr(i + ord('A')) for i in range(len(player_ids))])
    # player_names = np.array([str(i) for i in range(len(player_ids))])
    # I_rewards_v0 = []

    # log_rewards.cprint("-"*10 + " Rewards " + "-"*10)
    # for i in range(len(I_rewards)):
    #     if interaction_type == "opt":
    #         coalition = player_names[player_masks[i].astype(bool)].tolist()
    #     else:
    #         coalition = player_names[player_masks[i]].tolist()
    #     reward = I_rewards[i]
    #     I_rewards_v0.append(reward-I_rewards[0])
    #     if player_descriptions is None:
    #         log_str = f'v({" ".join(coalition)}): {reward}'
    #     else:
    #         if interaction_type == "opt":
    #             coalition_descriptions = player_descriptions[player_masks[i].astype(bool)].tolist()
    #         else:
    #             coalition_descriptions = player_descriptions[player_masks[i]].tolist()
    #         log_str = f'v({" ".join(coalition)}): {reward}\t ([{"][".join(coalition_descriptions)}])'
    #     log_rewards.cprint(log_str)
    # log_rewards.cprint("-"*10 + " Rewards-v() " + "-"*10)
    # and_order = np.argsort(-np.array(I_rewards_v0))[:log_nums]
    # if interaction_type != "opt":
    #     for i in and_order:
    #         coalition = player_names[player_masks[i]].tolist()
    #         reward_v0 = I_rewards[i]-I_rewards[0]
    #         print(f"reward:{I_rewards[i]},reward_v0:{reward_v0}")

    #         if player_descriptions is None:
    #             log_str1 = f'v({" ".join(coalition)}): {reward_v0}'
    #         else:
    #             coalition_descriptions = player_descriptions[player_masks[i]].tolist()
    #             log_str1 = f'v({" ".join(coalition)}): {reward_v0}, EP: {reward_v0/I_rewards_v0[-1]}\t ([{"][".join(coalition_descriptions)}])'
    #         log_rewards.cprint(log_str1)

    # log_rewards.cprint("="*50) # separator
    # log_rewards.close()


def log_inference(tokenizer, save_path, input_ids, pred_label, pred_scores):
    log_inference = LogWriter(os.path.join(save_path, "inference.txt"), verbose=True, write_mode='w')
    log_inference.cprint("Execution time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    log_inference.cprint(f"prompt: {tokenizer.decode(input_ids.squeeze())}")

    log_inference.cprint("-"*10 + " tokenizer " + "-"*10)
    for idx, input_id in enumerate(input_ids.squeeze()):
        log_inference.cprint(f"idx:{idx} input_id:{input_id} decoded_text: {tokenizer.decode([input_id])}")

    log_inference.cprint("-"*10 + " predict " + "-"*10)
    log_inference.cprint(f"pred_label:{pred_label}")
    log_inference.cprint(f"pred_scores:{pred_scores.tolist()}")
    log_inference.cprint(f"decoded_pred_text (only for generation task): {tokenizer.decode([pred_label])}")
    log_inference.cprint("="*50) # separator
    log_inference.close()


def are_words_in_same_sentence(tokenizer, text: Union[str, List[str]], word1: str, word2: str) -> bool:
    """Check if two words are in the same sentence

    Args:
        tokenizer: The tokenizer instance
        text: The full text containing both words (str or list of strings)
        word1: First word to check
        word2: Second word to check

    Returns:
        bool: True if words are in same sentence, False otherwise
    """
    # Convert list to string if needed
    if isinstance(text, list):
        text = ' '.join(text)

    # Split text into sentences using both comma and period as delimiters
    sentences = []
    current_sentence = []

    # Split text into tokens while preserving delimiters
    tokens = []
    for token in text.split():
        if token.endswith(('.', ',')):
            tokens.append(token[:-1])
            tokens.append(token[-1])
        else:
            tokens.append(token)

    # Reconstruct sentences
    for token in tokens:
        if token in ('.', ','):
            # End current sentence
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        else:
            current_sentence.append(token)

    # Add last sentence if exists
    if current_sentence:
        sentences.append(' '.join(current_sentence))

    # Find which sentences contain each word
    word1_sent_idx = None
    word2_sent_idx = None

    for i, sent in enumerate(sentences):
        # Decode tokens to text for comparison
        decoded_sent = tokenizer.decode(tokenizer(sent)['input_ids'])

        if word1 in decoded_sent:
            word1_sent_idx = i
        if word2 in decoded_sent:
            word2_sent_idx = i

    # If either word not found, assume they're in different sentences
    if word1_sent_idx is None or word2_sent_idx is None:
        return False

    # If both words found in same sentence
    return word1_sent_idx == word2_sent_idx

def log_generation(model, tokenizer, save_path, input_ids, attention_mask):
    log_inference = LogWriter(os.path.join(save_path, "generation.txt"), verbose=False, write_mode='w')
    log_inference.cprint("Execution time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    log_inference.cprint(f"prompt: {tokenizer.decode(input_ids.squeeze())}")

    log_inference.cprint("-"*10 + " generation " + "-"*10)
    generated_ids = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_new_tokens=50,
                                   use_cache=False)  # shape (batch_size=1, seq_len)
    generated_text = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=False)[0]
    generated_text_by_tokens = tokenizer.batch_decode(generated_ids[0], clean_up_tokenization_spaces=False)
    log_inference.cprint(f"generated_text: {generated_text}\n")
    log_inference.cprint("generated_text_by_tokens:")
    for token_text in generated_text_by_tokens:
        log_inference.cprint(f"{token_text}")
    log_inference.cprint("="*50) # separator
    log_inference.close()


class InteractionNLP:
    """
    Wrapper class to calculate the interaction
    """
    def __init__(self, calculator: Calculator, config: Dict, file_path_template,ep_count_file, single_feature_file, type, MP_size, MP_used_nodes):
        """
        config include the following keys (refer to run_interaction_nlp.py)
        - task: str, e.g., "nlp-seq-cls", "nlp-generation", "nlp-nli"
        - data_type: str, "float" or "double"
        - selected_dim: str, the dimension to calculate reward score
        - baseline_type: str, "learned", "unk", "pad", etc.
        - gt_type: str, "correct" or "predict"
        - background_type: str, "mask" or "ori"
        - sort_type: str, "order" or "binary"
        - cal_batch_size: int, the batch size for calculating the reward
        - verbose: bool/int, whether to print verbose information

        - sparse_mode: str, "pq", "p", "q", "none"
        - loss: str, "l1", "huber"
        - delta: float, the delta for huber loss
        - optimizer: str, "sgd", "adam"
        - lr: float, the learning rate
        - auto_lr: str, whether to use automatic learning rate
        - momentum: float, the momentum for optimizer (not used for adam)
        - niters: int, the number of iterations for optimization
        - qcoef: float, the coefficient for q bound
        - qstd: float, the standard for q bound
        - qscale: float, the scaling factor for q bound of different orders
        - qtricks: bool, whether to use q tricks (minus the mean of all q's at the end of each iteration)
        - piecewise: bool, not implemented
        - init_pq_path: str, the path to the initial p and q values for sparsification (optional)
        - mean_of_vN_v0: float, the mean of |vN - v0| (optional)

        """
        self.calculator = calculator
        self.config = config
        self.file_path_template = file_path_template
        self.ep_count_file = ep_count_file
        self.single_feature_file = single_feature_file
        self.type = type
        self.MP_size = MP_size
        self.MP_used_nodes = MP_used_nodes



    def __call__(self,
                 data_tuple: Dict,
                 player_ids: List[List],
                 save_path: str,
                 sample_id: int,
                 baseline_value: Union[torch.Tensor, None] = None) -> None:
        """
        :param data_tuple: Dict, keys include:
            - "input_ids": torch.LongTensor, shape (batch_size=1, seq_len)
            - "attention_mask": torch.LongTensor, shape (batch_size=1, seq_len)
            - "label": [optional] int, the label (not necessary if gt_type is "predict", and not necessarily the ground-truth label)
            - "sentence" or "text": str, the original input sentence
        :param player_ids: List[List], list of the player ids of the input sentence
            Example: [[0], [2, 3], [5], [7, 8, 9]]
        :param save_path: str, the path to save the results
        :param baseline_value: torch.Tensor (float/double) or None
            If not None, it is the vector of the (learned) baseline value embedding
            If it is None, it means the baseline value is specified by config["baseline_type"]:
                a certain token, e.g., <pad> or <unk>
        """
        os.makedirs(save_path, exist_ok=True)
        device = self.calculator.model.device

        input_ids = data_tuple["input_ids"].to(device) # shape (batch_size=1, seq_len)
        attention_mask = data_tuple["attention_mask"].to(device) # shape (batch_size=1, seq_len)

        # Get baseline value id / embeds
        if baseline_value is None:
            baseline_value = get_baseline_id_nlp(self.config["baseline_type"], self.calculator.tokenizer)
            assert isinstance(baseline_value, int), f"baseline_value should be an int, but got {type(baseline_value)}."
            baseline_value_embeds = self.calculator.get_embeds(torch.tensor(baseline_value, dtype=torch.long).to(device))
            # shape (embed_dim,)
        else:
            baseline_value_embeds = baseline_value.to(device)

        # Get the ground-truth or prediction label
        pred_label, pred_scores = get_pred_label_nlp(calculator=self.calculator,
                                        input_ids=input_ids,
                                        attention_mask=attention_mask)  # an int
        if self.config["gt_type"] == "correct":
            assert "label" in data_tuple, ("If gt_type is 'correct', the data_tuple should contain "
                                           "the target/correct/ground-truth label.")
            label = data_tuple["label"].squeeze().item()  # the label should be converted to an int
        elif self.config["gt_type"] == "predict":
            label = pred_label
        else:
            raise NotImplementedError(f"gt_type {self.config['gt_type']} not recognized.")

        # log inference results
        log_inference(self.calculator.tokenizer, save_path, input_ids, pred_label, pred_scores)

        # If it is a generation task, we let the model to generate the whole sentence, and then log it
        if self.config["task"] == "nlp-generation":
            log_generation(self.calculator.model, self.calculator.tokenizer, save_path, input_ids, attention_mask)

        # get the mask_input_function and forward_function for computing interactions
        if self.config["interaction_type"] == "opt":
            embedding_layer = self.calculator.model.get_input_embeddings()  # 获取 embedding 层
            baseline_embedding = torch.zeros(embedding_layer.embedding_dim,
                                             device=embedding_layer.weight.device)  # 例如全0向量

            mask_input_function = get_mask_input_function_nlp_opt(embedding_layer, baseline_embedding)

            forward_function = get_forward_function_nlp_opt(calculator=self.calculator,
                                                        attention_mask=attention_mask)
        else:
            mask_input_function = get_mask_input_function_nlp()

            forward_function = get_forward_function_nlp(calculator=self.calculator,
                                                        baseline_value_embeds=baseline_value_embeds,
                                                        attention_mask=attention_mask)


        background = [i for i in range(input_ids.shape[-1]) if i not in set(flatten(player_ids))]
        # serve as background variables, either staying masked or unmasked at all time

        if self.config["selected_dim"].startswith("gt-log-odds-sample="):
            num_sample_dims = int(self.config["selected_dim"].split("=")[-1])
            softmax_sample_dims = get_softmax_sample_dims_nlp(self.calculator,
                                                              input_ids,
                                                              attention_mask,
                                                              dimension_num=num_sample_dims).tolist()
        else:
            softmax_sample_dims = None

        if self.config.get("interaction_type", "harsanyi") == "shapley_taylor":
            interaction_runner = ShapleyTaylor(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,  # BASELINE_FLAG_NLP = -42
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
            )
        elif self.config.get("interaction_type") == "shapley_interaction_index":
            interaction_runner = ShapleyInteractionIndex(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
            )
        elif self.config.get("interaction_type") == "re":
            interaction_runner = CalculateReward(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
                file_path_template=self.file_path_template,
                ep_count_file = self.ep_count_file,
                single_feature_file = self.single_feature_file,
                type = self.type,
                MP_size = self.MP_size,
                MP_used_nodes =self.MP_used_nodes
            )
        elif self.config.get("interaction_type") == "opt":
            interaction_runner = CalculateOpt(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
                calculator=self.calculator,
                interaction_type=self.config["interaction_type"],
                file_path_template=self.file_path_template,
                ep_count_file = self.ep_count_file,
                single_feature_file = self.single_feature_file,
                type = self.type
            )
        elif self.config.get("interaction_type") == "shapley":
            interaction_runner = Shapley(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
            )
        elif self.config.get("interaction_type") == "shapleyC":
            interaction_runner = ShapleyC(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
            )
        elif self.config.get("interaction_type") == "gradient":
            print("Running gradient-based attribution...")
            self.calculator.model.eval()

            input_ids_grad = input_ids.clone().detach().to(device)
            attention_mask_grad = attention_mask.to(device)

            # 获取嵌入并开启梯度
            inputs_embeds = self.calculator.get_embeds(input_ids_grad).detach()
            inputs_embeds.requires_grad_(True)

            # 前向传播，获取目标 logit
            scores = self.calculator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_grad
            )  # (1, num_classes)
            selected_score = scores[0, label]

            # 反向传播
            self.calculator.model.zero_grad()
            selected_score.backward()

            grads = inputs_embeds.grad  # (1, seq_len, hidden_dim)

            # 对每个 player 计算梯度 norm（你也可以用别的聚合策略）
            player_grads = []
            for player in player_ids:
                grad_vector = grads[0, player, :]  # shape: (player_len, hidden_dim)
                grad_norm = torch.norm(grad_vector, dim=-1).mean().item()  # 平均 L2 norm
                player_grads.append(grad_norm)

            # 保存成和 interaction 一样的形式：每个 player 单独作为一个 coalition（只有1的掩码）
            n_players = len(player_ids)
            I_and = torch.tensor(player_grads)  # shape: (n_players,)
            player_masks = torch.eye(n_players, dtype=torch.bool)  # shape: (n_players, n_players)

            # 获取 player 的可读描述（词语）
            player_descriptions = get_player_words_from_ids(self.calculator.tokenizer, input_ids.squeeze(), player_ids)

            # 调用 log_interaction：注意这里只会打印单个 player 的“交互值”
            log_interaction(self.config["interaction_type"],save_path, player_ids, player_masks, I_and, player_descriptions)

            # 你可以（可选）保存梯度值成 .npy 文件
            np.save(osp.join(save_path, "player_grads.npy"), I_and.numpy())

            # ✅ 加上 return，避免后续还去跑 interaction_runner
            return
        elif self.config.get("interaction_type") == "integrated_gradients":
            print("Running integrated gradients attribution...")
            self.calculator.model.eval()

            dtype = next(self.calculator.model.parameters()).dtype
            device = input_ids.device

            print(f"[Debug] input_ids shape: {input_ids.shape}")
            print(f"[Debug] device: {device}, dtype: {dtype}")

            # 准备输入和baseline
            input_ids_grad = input_ids.clone().detach().to(device)
            attention_mask_grad = attention_mask.to(device)
            current_embeds = self.calculator.get_embeds(input_ids_grad).detach().to(dtype=dtype)
            print(f"[Debug] current_embeds shape: {current_embeds.shape}")

            if isinstance(baseline_value, int):
                baseline_ids = torch.full_like(input_ids_grad, baseline_value)
                baseline_embeds = self.calculator.get_embeds(baseline_ids).detach().to(dtype=dtype)
                print(f"[Debug] baseline_embeds (from id {baseline_value}) shape: {baseline_embeds.shape}")
            else:
                baseline_embeds = baseline_value_embeds.unsqueeze(0).unsqueeze(0).expand_as(current_embeds).to(
                    dtype=dtype)
                print(f"[Debug] baseline_embeds (custom) shape: {baseline_embeds.shape}")

            steps_total = self.config.get("ig_steps", 25)  # 总步数
            steps_per_batch = 5  # 每批步数
            print(f"[Debug] IG steps: total={steps_total}, per_batch={steps_per_batch}")

            total_grads_accum = torch.zeros_like(current_embeds)  # 累积梯度初始化

            for start_step in range(0, steps_total, steps_per_batch):
                end_step = min(start_step + steps_per_batch, steps_total)
                batch_steps = end_step - start_step
                print(f"[Debug] Processing steps {start_step}-{end_step} (batch size: {batch_steps})")

                # 计算该batch的alphas，线性均匀采样
                alphas = torch.linspace(
                    start_step / (steps_total - 1),
                    (end_step - 1) / (steps_total - 1),
                    batch_steps,
                    device=device,
                    dtype=dtype
                )
                print(f"[Debug] Alphas: {alphas.cpu().numpy()}")

                interpolated_embeds = baseline_embeds + alphas.view(-1, 1, 1) * (current_embeds - baseline_embeds)
                interpolated_embeds.requires_grad_(True)

                attention_mask_batch = attention_mask_grad.expand(batch_steps, -1)

                scores = self.calculator(
                    inputs_embeds=interpolated_embeds,
                    attention_mask=attention_mask_batch
                )
                selected_scores = scores[:, label]

                self.calculator.model.zero_grad()
                selected_scores.sum().backward()

                grads = interpolated_embeds.grad  # (batch_steps, seq_len, hidden_dim)
                batch_mean_grads = grads.mean(dim=0, keepdim=True)  # (1, seq_len, hidden_dim)
                print(f"[Debug] grads shape: {grads.shape}, batch_mean_grads shape: {batch_mean_grads.shape}")

                total_grads_accum += batch_mean_grads * batch_steps

            total_grads = total_grads_accum / steps_total
            ig = total_grads * (current_embeds - baseline_embeds)
            print(f"[Debug] Integrated gradients shape: {ig.shape}")

            player_importances = [ig[0, player, :].sum().item() for player in player_ids]
            print(f"[Debug] Player importances: {player_importances}")

            n_players = len(player_ids)
            I_and = torch.tensor(player_importances)
            player_masks = torch.eye(n_players, dtype=torch.bool)
            player_descriptions = get_player_words_from_ids(self.calculator.tokenizer, input_ids.squeeze(), player_ids)

            print(f"[Debug] Player descriptions: {player_descriptions}")
            print(f"[Debug] Saving results to: {save_path}")

            log_interaction("ig", save_path, player_ids, player_masks, I_and, player_descriptions)
            np.save(osp.join(save_path, "player_integrated_gradients.npy"), I_and.numpy())

            print("[Debug] Integrated gradients attribution completed.")
            return
        else:
            interaction_runner = AndOrHarsanyi(
                forward_function=forward_function,
                selected_dim=self.config["selected_dim"],
                x=input_ids,
                baseline=BASELINE_FLAG_NLP,  # BASELINE_FLAG_NLP = -42
                y=label,
                sample_id=sample_id,
                all_players_subset=player_ids,
                background=background,
                background_type=self.config["background_type"],
                mask_input_function=mask_input_function,
                cal_batch_size=self.config["cal_batch_size"],
                softmax_sample_dims=softmax_sample_dims,
                sort_type=self.config["sort_type"],
                verbose=self.config["verbose"],
            )

        #interaction_runner.compute_interactions()
        if self.config.get("interaction_type", "harsanyi") == "shapley_taylor":
            interaction_runner.compute_interactions()
            I_shapley = interaction_runner.get_shapley_interaction()
            np.save(osp.join(save_path, "I_shapley.npy"), I_shapley.cpu().numpy())
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.cpu().numpy())
            np.save(osp.join(save_path, "sample_masks.npy"), interaction_runner.sample_masks.cpu().numpy())
            I_and = I_shapley
            interaction_runner.save(save_path)
            #I_or = I_shapley
        elif self.config.get("interaction_type") == "shapley_interaction_index":
            interaction_runner.compute_interactions()
            I_shapley = interaction_runner.get_shapley_interaction_index()
            np.save(osp.join(save_path, "I_shapley.npy"), I_shapley.cpu().numpy())
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.cpu().numpy())
            np.save(osp.join(save_path, "sample_masks.npy"), interaction_runner.sample_masks.cpu().numpy())
            I_and = I_shapley
            interaction_runner.save(save_path)
            #I_or = I_shapley
        elif self.config.get("interaction_type") == "re":
            print("re")
            interaction_runner.compute_interactions()
            interaction_runner.save(save_path)
            I_and = interaction_runner.get_rewards()
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.cpu().numpy())
        elif self.config.get("interaction_type") == "opt":
            print("opt")
            interaction_runner.compute_interactions(k_file=self.config["ep_count_file"], tau_start=self.config["tau_start"], tau_end=self.config["tau_end"], tau_rate=self.config["tau_rate"],opt_path=self.config["opt_path"], sample_id=sample_id, baseline_value_embeds=baseline_value_embeds,attention_mask=attention_mask)
            interaction_runner.save(save_path)
            I_and = interaction_runner.get_rewards()
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.detach().cpu().numpy())
        elif self.config.get("interaction_type") == "shapley":
            interaction_runner.compute_interactions()
            I_shapleyv = interaction_runner.get_shapley_value()
            np.save(osp.join(save_path, "I_shapleyv.npy"), I_shapleyv.cpu().numpy())
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.cpu().numpy())
            np.save(osp.join(save_path, "sample_masks.npy"), interaction_runner.sample_masks.cpu().numpy())
            I_and = I_shapleyv
            interaction_runner.save(save_path)
            #I_or = I_shapley
        elif self.config.get("interaction_type") == "shapleyC":
            interaction_runner.compute_interactions()
            I_shapleyv = interaction_runner.get_shapley_value()
            np.save(osp.join(save_path, "I_shapleyv.npy"), I_shapleyv.cpu().numpy())
            np.save(osp.join(save_path, "player_masks.npy"), interaction_runner.player_masks.cpu().numpy())
            #np.save(osp.join(save_path, "sample_masks.npy"), interaction_runner.sample_masks.cpu().numpy())
            I_and = I_shapleyv
            interaction_runner.save(save_path)
            #I_or = I_shapley
        else:
            print("No interaction type")
            interaction_runner.compute_interactions()
            interaction_runner.save(save_path)
            I_and = interaction_runner.get_and_interaction()
            #I_or = interaction_runner.get_or_interaction()
        player_masks = interaction_runner.get_player_masks()

        # Filter out interactions between words in different sentences
        text = data_tuple.get("sentence") or data_tuple.get("text")
        if text:
            # Get player words and their positions
            player_words = get_player_words_from_ids(self.calculator.tokenizer, input_ids.squeeze(), player_ids)

            # Create mask for same-sentence word pairs
            n_players = len(player_words)
            player_sentences = []

            # Precompute sentence membership for each player
            for words in player_words:
                if words:
                    # Use first word to determine sentence membership
                    player_sentences.append(are_words_in_same_sentence(
                        self.calculator.tokenizer, text, words[0], words[0]
                    ))
                else:
                    player_sentences.append(True)  # Empty players stay in same sentence

            # Create mask for player combinations
            n_combinations = I_and.shape[0]
            same_sentence_mask = torch.ones(n_combinations, dtype=torch.bool)

            # Get player masks for each combination
            player_masks = interaction_runner.get_player_masks()

            # Check sentence membership for each combination
            for i in range(n_combinations):
                # Get indices of players in this combination
                players_in_combo = [j for j, mask in enumerate(player_masks[i]) if mask]

                # Check if all players are in the same sentence
                if len(players_in_combo) > 1:
                    first_sentence = player_sentences[players_in_combo[0]]
                    for player_idx in players_in_combo[1:]:
                        if player_sentences[player_idx] != first_sentence:
                            same_sentence_mask[i] = False
                            break

            #Apply mask to interactions
            # if I_and is not None:
            #     I_and = I_and * same_sentence_mask.to(I_and.device)
            # if I_or is not None:
            #     I_or = I_or * same_sentence_mask.to(I_or.device)
        I_rewards = interaction_runner.get_rewards()

        player_descriptions = get_player_words_from_ids(self.calculator.tokenizer, input_ids.squeeze(), player_ids)
        if self.config.get("interaction_type", "harsanyi") == "shapley_taylor":
            log_interaction(save_path, player_ids, player_masks, I_and, player_descriptions)
        else:
            log_interaction(self.config["interaction_type"],save_path, player_ids, player_masks, I_and, player_descriptions)
        log_rewards(self.config["interaction_type"],save_path, player_ids, player_masks, I_rewards, player_descriptions)
