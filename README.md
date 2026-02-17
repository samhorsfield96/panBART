
# PanBART, a pangenome long-context encoder-decoder attention based-architecture and associated analysis scripts.

## Generating input data

To generate input data, use [WTBCluster](https://github.com/samhorsfield96/WTBcluster). This will generate data in the ```tokenized_genomes``` directory.

## Training the model

To train panBART, use the ```panBART.py``` script, specifying required hyperparameters:

```
python panBART.py --input_file /path/to/tokenized/genomes.txt --model_save_path output.chk --attention_window 512 --log_dir path/to/log/dir --batch_size 1 --max_seq_length 8192 --num_heads 8 --num_layers 8 --embed_dim 256 --tokenizer_path /path/to/tokenize.json --epochs 400 --prop_masked 0.15 --num_workers 1 --learning_rate 0.000001 --early_stop_patience 10 --min_delta 0.001 --model_dropout_rate 0.4
```

If using ```panBART_deepspeed.py```, first generate the tokenizer:

```
panBART_deepspeed.py --train_file /path/to/training/genomes.txt --val_file /path/to/validation/genomes.txt --test_file /path/to/testing/genomes.txt --model_save_path output_deepspeed.chk --attention_window 512 --log_dir deepspeed_log --batch_size 1 --max_seq_length 7200 --num_heads 8 --num_layers 8 --embed_dim 256 --tokenizer_path tokenizer_deepspeed.json --epochs 300 --prop_masked 0.15 --num_workers 1 --early_stop_patience 10 --min_delta 0.01 --model_dropout_rate 0.4 --deepspeed_config deepspeed_ZS2_fp16.json --generate_tokenizer
```

Then run deepspeed training:

```
deepspeed panBART_deepspeed.py --train_file /path/to/training/genomes.txt --val_file /path/to/validation/genomes.txt --test_file /path/to/testing/genomes.txt --model_save_path output_deepspeed.chk --attention_window 512 --log_dir deepspeed_log --batch_size 1 --max_seq_length 7200 --num_heads 8 --num_layers 8 --embed_dim 256 --tokenizer_path tokenizer_deepspeed.json --epochs 300 --prop_masked 0.15 --num_workers 1 --early_stop_patience 10 --min_delta 0.01 --model_dropout_rate 0.4 --deepspeed_config deepspeed_ZS2_fp16.json
```