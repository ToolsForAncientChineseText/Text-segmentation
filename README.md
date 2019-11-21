# 古文断句
基于Google的BERT模型，为古籍自动添加断句。使用滑动窗口式古文断句方法，输入的长度不受限制。
详情请参考《基于BERT的古文断句研究与应用》（ http://jcip.cipsc.org.cn/CN/abstract/abstract2861.shtml ）
# 模型参数

# 运行
```
python para_seg.py \
  --max_prediction=2 \
  --input_file=./data/sample.txt \
  --output_file=./output/ \
  --vocab_file=./vocab.txt \
  --bert_config_file=./bert_config.json \
  --init_checkpoint=./checkpoint/model.ckpt \
  --max_seq_length=64
```
# 参考
https://github.com/google-research/bert
