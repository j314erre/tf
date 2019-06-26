# tf
tensorflow NLP experiments

## neural discourse

### training

`python tensorflow_chat2b.py --input /deeplearning/data/opensubtitles2016/opensubtitles2016_context10.txt --model_dir /deeplearning/models/chat2b_window5o/ --size=512 --num_layers=1 --steps_per_checkpoint=10000 --dev_train_split=0.2 --max_context_window=5`

### testing

`python/tensorflow_chat2b.py **--decode** --input /deeplearning/data/opensubtitles2016/opensubtitles2016_context10.dev.txt --model_dir /home/jpierre/deeplearning/models/chat2b_window5o/ --size=512 --num_layers=1 --steps_per_checkpoint=10000 --dev_train_split=0.2 --max_context_window=5`

## translation / chatbot

### training

`python tensorflow_translate.py --input /deeplearning/twitter_msft/msft_twitter_218k.train.txt --model_dir /deeplearning/models/msft1/ --size=512 --num_layers=2 --steps_per_checkpoint=200 --dev_train_split=0.01`

## RNN text classifiation

### training

` python tensorflow_rnn_classify.py --input=/deeplearning/data/twitter_fortune500/fortune500_twitter.train_classify.txt --model_dir=/deeplearning/models/twtr_rnn_classify_lstm/ --embedding_size=512 --source_vocab_size=40000 --num_classes=2 --dev_train_split=0.2 --steps_per_checkpoint=500 --max_global_step=5000 --use_lstm`

## 
