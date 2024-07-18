import torch
from torch.utils.data import Dataset

from utils import neg_sample, Substitute


class SeqDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.SSE = Substitute(args.item_similarity_model, args.ss_probability)

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test", "pretrain"}

        if self.data_type == "train":
            s = items[:-2]
            s = self.SSE(s)
            input_ids = s[:-1]
            target_pos = s[1:]
            answer = [0]  # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # if user sequence length < max_len，add 0 on the front-end
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg
        # if user sequence length > max_len，cut the last max_len
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long)
            )
        else: # all of shape: b*max_sq
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long), # training
                torch.tensor(target_pos, dtype=torch.long), # targeting, one item right-shifted, since task is to predict next item
                torch.tensor(target_neg, dtype=torch.long), # random sample an item out of training and eval for every training items.
                torch.tensor(answer, dtype=torch.long) # last item for prediction.
            )
        return cur_tensors

    def __len__(self):
        return len(self.user_seq)