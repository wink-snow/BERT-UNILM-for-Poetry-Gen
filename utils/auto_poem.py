from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.tokenizers import Tokenizer
from typing import Literal, Dict, List, Optional, Any, Tuple
import torch

from utils.const import convert_rhyme_dict, POEM_STRUCTURES, PENALTIES_REWARDS

CUSTOM_RHYME_DICT = convert_rhyme_dict()

class AutoPoem(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def __init__(
            self, 
            model,
            tokenizer: Tokenizer,
            bos_token_id:int=None, 
            eos_token_id:int=-1, 
            max_new_tokens:int=None, 
            min_new_tokens:int=1, 
            max_length:int=128, 
            pad_token_id:int=0, 
            padding_side:Literal['left', 'right']='right', 
            device:str='cpu', 
            n:int=1, 
            top_k:int=None, 
            top_p:float=None,
            temperature:float=1.0, 
            repetition_penalty:float=1.0, 
            min_ends:int=1, 

            poem_type: Literal['jueju_5', 'jueju_7', 'lishi_5', 'lishi_7'] = 'jueju_5',
            rhyme_dict: Dict[str, str] = None,
            custom_generation_config: Dict[str, float] = None,
            **generation_config
        ):
        super().__init__(
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, 
            max_new_tokens=max_new_tokens, 
            min_new_tokens=min_new_tokens, 
            max_length=max_length, 
            pad_token_id=pad_token_id, 
            padding_side=padding_side, 
            device=device, 
            n=n, 
            top_k=top_k, 
            top_p=top_p,
            temperature=temperature, 
            repetition_penalty=repetition_penalty, 
            min_ends=min_ends,
            **generation_config
        )
        self.model = model
        self.tokenizer = tokenizer
        self.poem_type = poem_type
        self.poem_struct = POEM_STRUCTURES.get(self.poem_type)
        
        self.chars_per_line = self.poem_struct['chars_per_line']
        self.total_lines = self.poem_struct['lines']
        self.rhyming_lines_idx = self.poem_struct['rhyme_lines_idx']

        self.rhyme_dict = rhyme_dict if rhyme_dict else CUSTOM_RHYME_DICT

        self.comma_id = tokenizer.token_to_id('，') if '，' in tokenizer._token_dict else -100
        self.period_id = tokenizer.token_to_id('。') if '。' in tokenizer._token_dict else -100
        self.punc_ids = [self.comma_id, self.period_id]

        default_penalties_rewards = PENALTIES_REWARDS
        self.penalties_rewards = {**default_penalties_rewards, **(custom_generation_config or {})}
        
    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
    def predict(self, inputs: List[torch.Tensor], output_ids: torch.Tensor, states: Optional[List[Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        token_ids_source, segment_ids_source = inputs  # inputs: [CLS] text_a [SEP]

        true_generated_ids = output_ids[:, 1:] 

        token_ids_combined = torch.cat([token_ids_source, true_generated_ids], 1)
        
        segment_ids_generated = torch.ones_like(true_generated_ids, device=segment_ids_source.device, dtype=segment_ids_source.dtype)
        segment_ids_combined = torch.cat([segment_ids_source, segment_ids_generated], 1)
        
        _, y_pred = self.model.predict([token_ids_combined, segment_ids_combined])
        logits = y_pred[:, -1, :]

        num_beams = output_ids.shape[0]
        
        if states is None:
            _processed_states = [{} for _ in range(num_beams)]
        elif len(states) != num_beams:
            if len(states) < num_beams:
                _processed_states = states + [{} for _ in range(num_beams - len(states))]
            else: 
                _processed_states = states[:num_beams]
        else:
            _processed_states = [s if s is not None else {} for s in states]

        new_states_list = []

        for i in range(output_ids.shape[0]):
            current_beam_output_ids_list = true_generated_ids[i].tolist()

            current_beam_state = _processed_states[i]
            first_rhyme_group = current_beam_state.get('first_rhyme_group', None)
            first_rhyme_char_id = current_beam_state.get('first_rhyme_char_id', None)

            current_line_index = 0
            current_char_index_in_line = 0
            # temp_first_rhyme_group = first_rhyme_group 
            # temp_first_rhyme_char_id = first_rhyme_char_id

            _temp_line_idx = 0
            _temp_char_idx = 0

            for token_id in current_beam_output_ids_list:
                if token_id in self.punc_ids:
                    # if temp_first_rhyme_group is None and \
                    #    _temp_line_idx in self.rhyming_lines_idx and \
                    #    _temp_char_idx == self.chars_per_line:
                    #     idx_of_last_char = -1
                    
                    _temp_line_idx += 1
                    _temp_char_idx = 0
                else:
                    _temp_char_idx += 1
            
            current_line_index = _temp_line_idx
            current_char_index_in_line = _temp_char_idx

            title_first_char_id = token_ids_source[i, 1].item() 
            prev_char_id = current_beam_output_ids_list[-1] if current_beam_output_ids_list else -1

            for token_candidate_id in range(logits.shape[1]): 
                candidate_char = self.tokenizer.id_to_token(token_candidate_id)
                if candidate_char is None: continue 

                if not current_beam_output_ids_list and token_candidate_id == title_first_char_id:
                    logits[i, token_candidate_id] -= self.penalties_rewards["first_char_vs_title_penalty"]

                if token_candidate_id in current_beam_output_ids_list and token_candidate_id not in self.punc_ids:
                    allow_die_ci = False
                    if self.poem_struct['lines'] == 8 and \
                       current_line_index in [2, 3, 4, 5] and \
                       current_char_index_in_line > 0 and \
                       token_candidate_id == prev_char_id: 
                        allow_die_ci = True
                    
                    if allow_die_ci:
                        logits[i, token_candidate_id] += self.penalties_rewards["die_ci_reward"] 
                    else:
                        logits[i, token_candidate_id] -= self.penalties_rewards["repeat_char_penalty"]

                if (current_char_index_in_line + 1) == self.chars_per_line:
                    if current_line_index in self.rhyming_lines_idx: 
                        candidate_rhyme_group = self.rhyme_dict.get(candidate_char)
                        
                        if candidate_rhyme_group is not None: 
                            if first_rhyme_group is None: 
                                logits[i, token_candidate_id] += self.penalties_rewards["rhyme_reward"]
                                logits[i, token_candidate_id] += self.penalties_rewards["rhyme_bonus_if_in_dict"]
                            elif candidate_rhyme_group == first_rhyme_group: 
                                logits[i, token_candidate_id] += self.penalties_rewards["rhyme_reward"]
                            else: 
                                logits[i, token_candidate_id] -= self.penalties_rewards["wrong_rhyme_penalty"]
                        else: 
                            logits[i, token_candidate_id] -= self.penalties_rewards["not_rhyming_at_rhyme_pos_penalty"]
                
                if current_char_index_in_line == self.chars_per_line:
                    is_last_line = (current_line_index == self.total_lines - 1)
                    target_punc_id = self.period_id if is_last_line else self.comma_id
                    
                    if token_candidate_id == target_punc_id:
                        logits[i, token_candidate_id] += 5.0 
                    elif token_candidate_id in self.punc_ids: 
                        logits[i, token_candidate_id] -= 5.0 
                    else: 
                        logits[i, token_candidate_id] -= 10.0 

            new_beam_state = {'first_rhyme_group': first_rhyme_group, 'first_rhyme_char_id': first_rhyme_char_id}
            if new_beam_state['first_rhyme_group'] is None and current_beam_output_ids_list:
                _s_line_idx_of_last_char = 0
                _s_char_idx_in_line_of_last_char = 0
                last_non_punc_token_id = -1

                for _tok_id in current_beam_output_ids_list:
                    if _tok_id in self.punc_ids:
                        if last_non_punc_token_id != -1 and _s_char_idx_in_line_of_last_char == self.chars_per_line:
                            
                            if _s_line_idx_of_last_char in self.rhyming_lines_idx:
                                last_char_token = self.tokenizer.id_to_token(last_non_punc_token_id)
                                if last_char_token:
                                    rhyme_group_of_last = self.rhyme_dict.get(last_char_token)
                                    if rhyme_group_of_last and new_beam_state['first_rhyme_group'] is None: # 确保只设置一次
                                        new_beam_state['first_rhyme_group'] = rhyme_group_of_last
                                        new_beam_state['first_rhyme_char_id'] = last_non_punc_token_id
                                        
                                        break # 
                        
                        _s_line_idx_of_last_char +=1
                        _s_char_idx_in_line_of_last_char = 0
                        last_non_punc_token_id = -1
                    else:
                        _s_char_idx_in_line_of_last_char +=1
                        last_non_punc_token_id = _tok_id
                
                if new_beam_state['first_rhyme_group'] is None and \
                   last_non_punc_token_id != -1 and \
                   _s_char_idx_in_line_of_last_char == self.chars_per_line and \
                   _s_line_idx_of_last_char in self.rhyming_lines_idx:
                    last_char_token = self.tokenizer.id_to_token(last_non_punc_token_id)
                    if last_char_token:
                        rhyme_group_of_last = self.rhyme_dict.get(last_char_token)
                        if rhyme_group_of_last:
                            new_beam_state['first_rhyme_group'] = rhyme_group_of_last
                            new_beam_state['first_rhyme_char_id'] = last_non_punc_token_id
            
            new_states_list.append(new_beam_state)
        
        return logits, new_states_list

    def generate(self, text, top_k=1, top_p=0.95, temperature=1.0): 
        max_c_len = self.max_length - self.max_new_tokens
        # tokenizer.encode(text) -> [CLS] text [SEP]
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=max_c_len) 
        token_ids_tensor = torch.tensor([token_ids], device=self.device)
        segment_ids_tensor = torch.tensor([segment_ids], device=self.device)
        
        search_results = self.beam_search(inputs=[token_ids_tensor, segment_ids_tensor], top_k=top_k, top_p=top_p, temperature=temperature)
        
        # if isinstance(search_results, tuple) and len(search_results) == 2:
        #     output_ids = search_results[0][0] 
        # else:
        #     output_ids = search_results[0] 
        if isinstance(search_results, tuple) and len(search_results) == 2:
            output_ids = search_results[0][0] 
        elif isinstance(search_results, list) and len(search_results) > 0:
            output_ids = search_results[0]
        else: 
            output_ids = search_results[0] if isinstance(search_results, torch.Tensor) and search_results.ndim > 1 else search_results

        return self.tokenizer.decode(output_ids.cpu().numpy())