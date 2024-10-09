import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, cal_mrr


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model

        if self.cuda_condition:
            self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True) 
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)
    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)
    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)
    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], 0
        for k in [1, 5, 10, 20, 40]:
            recall_result = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            ndcg_result = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
        mrr = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(recall[0]),
            "HIT@5": '{:.4f}'.format(recall[1]), "NDCG@5": '{:.4f}'.format(ndcg[1]),
            "HIT@10": '{:.4f}'.format(recall[2]), "NDCG@10": '{:.4f}'.format(ndcg[2]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
            "HIT@40": '{:.4f}'.format(recall[4]), "NDCG@40": '{:.4f}'.format(ndcg[4]),
            "MRR": '{:.4f}'.format(mrr)
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], mrr], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids) 
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_out = seq_out - self.model.position_embeddings.weight
        seq_emb = seq_out.reshape(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        auc = torch.sum(((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget) / torch.sum(istarget)
        return loss, auc

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        seq_out = seq_out - self.model.position_embeddings.weight[-1]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class FDT4SRTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(FDT4SRTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)
        self.args = args

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        rec_data_iter = dataloader
        if train:
            self.args.add_noise = True
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0
            for batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch

                reconstructed_seq = self.model.forward(input_ids)
                loss, recons_auc = self.cross_entropy(reconstructed_seq, target_pos, target_neg)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += recons_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
                }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.args.add_noise = False
            self.model.eval()
            with torch.no_grad():
                pred_list = None
                if full_sort:
                    answer_list = None
                    #for i, batch in rec_data_iter:
                    i = 0
                    for batch in rec_data_iter:
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch

                        recommend_reconstruct = self.model.forward(input_ids)

                        recommend_output = recommend_reconstruct[:, -1, :]
                        rating_pred = self.predict_full(recommend_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()

                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                        ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1

                    return self.get_full_sort_score(epoch, answer_list, pred_list)

                else:
                    assert "We need full_sort evaluation"    
                    return 0



