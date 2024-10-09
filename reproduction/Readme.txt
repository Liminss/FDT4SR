Sports:	
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.7
  ss_probability：0.05
  lr：0.0001
  batch_szie:128

Beauty:
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.7
  hidden_dropout_prob：0.3
  ss_probability：0.05
  lr：0.0001
  batch_szie:128

Tools:
  hidden_size：128
  inner_size: 512
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.5
  ss_probability：0.08
  lr：0.0001
  batch_szie:128

LastFM:
  hidden_size：128
  inner_size: 256
  attention_probs_dropout_prob：0.5
  hidden_dropout_prob：0.5
  ss_probability：0.07
  lr：0.001
  batch_szie:256

Hardware
CPU: 12 vCPU Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
GPU: RTX 3080 Ti(12GB) 

Software
Python 3.8.16
Pytorch 1.13.1
