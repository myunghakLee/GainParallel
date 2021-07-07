rm logs/train_GAIN_BERT_base.log
./run_GAIN_BERT.sh 0
tail -f -n 2000 logs/train_GAIN_BERT_base.log
