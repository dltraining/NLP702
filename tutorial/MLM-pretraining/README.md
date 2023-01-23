# Run Language Model Pretraining

This code was tested on:
* python/3.6.8
* troch==1.8.0
* transformers==4.6.0
* datasets==1.11.0

We can use RoBERTa checkpoint from here: https://huggingface.co/roberta-base
* wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
* wget https://huggingface.co/roberta-base/resolve/main/config.json
* wget https://huggingface.co/roberta-base/resolve/main/dict.txt
* wget https://huggingface.co/roberta-base/resolve/main/merges.txt
* wget https://huggingface.co/roberta-base/resolve/main/tokenizer.json
* wget https://huggingface.co/roberta-base/resolve/main/vocab.json


## Instruction
* Training data: training_sample.txt. Each sample is in a line. Samples are split by a empty line.
* Job shell script: run.sh
* Input arguments: Line 499-670
* Load model and tokenizer checkpoints. If a checkpoint cannot be found, a new model will be initialized: Line 747-763
* Data loader: Line 69-94 and Line 107-165
* Training loop: Line 244-439
* Mask input tokens: Line 210-241
* Evaluation loop: Line 442-495
* Save checkpoint: Line 825-840
