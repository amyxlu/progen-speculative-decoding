from collections import UserDict
import torch

class D(UserDict):
    def to(self, device):
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        return self
    def __getattr__(self, key):
        return self.data[key]
        

def pad_batch(input_ids, pad_token_id, padding_side='left'):
    '''
    Args:
        input_ids: List[List[int]]
        pad_token_id: int
        padding_side: 'left' | 'right'
    Returns:
        input_ids: Tensor
        attention_mask: tensor
    '''
    max_len = max(len(ls) for ls in input_ids)
    if padding_side == 'left':
        padded = [
            (max_len - len(seq)) * [pad_token_id] + seq
            for seq in input_ids
        ]
        attention_mask = [
            (max_len - len(seq)) * [0] + len(seq) * [1]
            for seq in input_ids
        ]
    else:
        padded = [
            seq + (max_len - len(seq)) * [pad_token_id]
            for seq in input_ids
        ]
        attention_mask = [
            len(seq) * [1] + (max_len - len(seq)) * [pad_token_id]
            for seq in input_ids
        ]
    return D(**{
        'input_ids': torch.tensor(padded, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    })
    