
# PanBART, a pangenome long-context encoder-decoder attention based-architecture and associated analysis scripts.

Developed by: Samuel Horsfield, initially forked from [panGPT](https://github.com/mol-evol/panGPT) developed by James McInerney

## Generating input data

To generate input data, use [WTBCluster](https://github.com/samhorsfield96/WTBcluster). This will generate data in the ```tokenized_genomes``` directory.

## Training the model

To train panBART, use the ```panGPT.py``` script, specifying required hyperparameters:

```
python panGPT.py --input_file /path/to/tokenized/genomes.txt --model_save_path output.chk --attention_window 512 --log_dir path/to/log/dir --batch_size 1 --max_seq_length 8192 --num_heads 8 --num_layers 8 --embed_dim 256 --tokenizer_path /path/to/tokenize.json --epochs 400 --prop_masked 0.15 --num_workers 1 --learning_rate 0.000001 --early_stop_patience 10 --min_delta 0.001 --model_dropout_rate 0.4
```