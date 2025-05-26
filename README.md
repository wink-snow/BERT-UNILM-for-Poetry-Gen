# BERT-UNILM-for-Poetry-Gen

This is a Poetry Generation project based on BERT-UNILM. We use the `chinese-bert-wwm` as pre-trained model. You could get the pre-trained model from [here](https://huggingface.co/hfl/chinese-bert-wwm/tree/main).

Welcome to star this repo if you like it!:dizzy:

## :rocket: Environment

- Pytorch
- Datasets: [chinese-poetry](https://github.com/Werneror/Poetry)

## :fire: Feature

- [x] Generate Chinese poems with different styles
- [x] Strength the `beam_search` method
- [ ] Deploy the model to the web

## :hammer: Structure
```bash
├─data
│  ├─processed
│  └─raw
├─pretrained_weights
│  └─bert_chinese_wwm
├─scripts                   # Scripts for processing data
|  └─process_data.py
├─test                      # Test data dir when debugging
│  └─data
├─utils                     # Utils for training and testing
│  └─auto_poem.py
|  └─const.py
|  └─load_data.py
├─cipai.txt
├─task_seq2seq_auto_poem.py # Main script for training
├─test_seq2seq.py           # Main script for testing
├─requirements.txt
├─README.md
```