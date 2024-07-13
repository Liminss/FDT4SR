import os
import numpy as np
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SeqDataset
from trainers import FDT4SRTrainer
from models import FDT4SR
from modules import ItemSimilarity
from utils import EarlyStopping, get_user_seqs, check_path, set_seed


def main():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='LastFM', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, 
                        help="Method to generate item similarity score. choices: ItemCF, ItemCF_IUF(Inverse user frequency)")

    # model args
    parser.add_argument("--model_name", default='FDT4SR', type=str)
    parser.add_argument("--hidden_size", type=int, default=200, help="hidden size of transformer model")
    parser.add_argument("--inner_size", type=int, default=400, help="2 or 4 * hidden_size")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="relu", type=str)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.3, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--use_position_embeddings', default=True, type=bool)
    parser.add_argument('--use_order', default=False, type=bool)
    parser.add_argument('--use_distance', default=False, type=bool)
    parser.add_argument("--ss_probability", type=float, default=0.05)

    # train args
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print result")
    parser.add_argument("--seed", default=816, type=int)
    parser.add_argument("--patience", default=30, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = get_user_seqs(args.data_file)

    args.item_size = max_item + 1
    args.num_users = num_users

    args.train_matrix = valid_rating_matrix
    print(f"valid rating matix shape: {valid_rating_matrix.shape}")

    # save args
    cur_time = datetime.now().strftime('%m-%d_%H-%M')
    args_str = f'{args.model_name}' \
               f'-{args.data_name}' \
               f'-{args.hidden_size}' \
               f'-{args.num_hidden_layers}' \
               f'-{args.num_attention_heads}' \
               f'-{args.hidden_act}' \
               f'-{args.attention_probs_dropout_prob}' \
               f'-{args.hidden_dropout_prob}' \
               f'-{args.max_seq_length}' \
               f'-{args.lr}' \
               f'-{args.weight_decay}'\
               f'-{cur_time}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')

    args.similarity_model_path = os.path.join(args.data_dir, args.data_name+'_'+args.similarity_model_name+'_similarity.pkl')
    args.item_similarity_model = ItemSimilarity(data_file=args.data_file, similarity_path=args.similarity_model_path,
                                                        model_name=args.similarity_model_name, dataset_name=args.data_name)

    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SeqDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SeqDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = FDT4SR(args=args)
    trainer = FDT4SRTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        # load the best model
        print('---------------load best model and do eval-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path)) # "./reproduction/LastFM.pt"
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info = trainer.test('best', full_sort=True)
    else:
        print('Start training......')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array([scores[3], scores[4]]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _ = trainer.valid('valid', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info = trainer.test('test', full_sort=True)

    print(args_str)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

if __name__ == '__main__':
    main()
