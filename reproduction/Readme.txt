Sports:	
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.7
  ss_probability：0.05
  lr：0.0001
  batch_szie:128
Total Parameters: 2611300
---------------load best model and do eval-------------------
{'Epoch': 'best', 'HIT@1': '0.0123', 'HIT@5': '0.0331', 'NDCG@5': '0.0228', 'HIT@10': '0.0481', 'NDCG@10': '0.0276', 'HIT@20': '0.0711', 'NDCG@20': '0.0334', 'HIT@40': '0.1029', 'NDCG@40': '0.0399', 'MRR': '0.0240'}

Beauty:
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.7
  hidden_dropout_prob：0.3
  ss_probability：0.05
  lr：0.0001
  batch_szie:128
Total Parameters: 1810532
---------------load best model and do eval-------------------
{'Epoch': 'best', 'HIT@1': '0.0193', 'HIT@5': '0.0521', 'NDCG@5': '0.0359', 'HIT@10': '0.0762', 'NDCG@10': '0.0436', 'HIT@20': '0.1095', 'NDCG@20': '0.0520', 'HIT@40': '0.1523', 'NDCG@40': '0.0608', 'MRR': '0.0376'}

Tools:
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.5
  ss_probability：0.08
  lr：0.0001
  batch_szie:128
Total Parameters: 1569380
---------------load best model and do eval-------------------
{'Epoch': 'best', 'HIT@1': '0.0139', 'HIT@5': '0.0344', 'NDCG@5': '0.0243', 'HIT@10': '0.0511', 'NDCG@10': '0.0297', 'HIT@20': '0.0745', 'NDCG@20': '0.0355', 'HIT@40': '0.1027', 'NDCG@40': '0.0413', 'MRR': '0.0257'}

LastFM:
  hidden_size：128
  inner_size: 256
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.5
  ss_probability：0.07
  lr：0.001
  batch_szie:256
Total Parameters: 662500
---------------load best model and do eval-------------------
{'Epoch': 'best', 'HIT@1': '0.0202', 'HIT@5': '0.0633', 'NDCG@5': '0.0421', 'HIT@10': '0.0936', 'NDCG@10': '0.0517', 'HIT@20': '0.1294', 'NDCG@20': '0.0608', 'HIT@40': '0.1798', 'NDCG@40': '0.0710', 'MRR': '0.0432'}
