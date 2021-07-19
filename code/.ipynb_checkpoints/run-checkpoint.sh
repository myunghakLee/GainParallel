# +
MODE=$1
RE_lr_scale=$2

rm logs/train_GAIN_BERT_base${MODE}_${RE_lr_scale}.log
./run_GAIN_BERT.sh 0 ${MODE} ${RE_lr_scale}
sudo chmod 777 logs/train_GAIN_BERT_base${MODE}_${RE_lr_scale}.log
tail -f -n 2000 logs/train_GAIN_BERT_base${MODE}_${RE_lr_scale}.log
