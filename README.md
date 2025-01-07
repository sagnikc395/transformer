## transformer

My implementation of the transformer architecture,
after reading [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

### Structure:

```
./
├── README.md
├── requirements.txt
├── ruff.toml
└── src
    ├── config.py
    ├── dataset.py
    ├── model.py
    └── train.py

2 directories, 7 files
```

### Contents:

- README.md
  > this file (duh!)
- src/model.py
  > has the basic structure of our transformer model.
- src/train.py
  > has utiltiies for getting the data from huggingface, building out the tokenizer , getting all sentences and building the vocabulary and then training the model on top our architecture.
- src/config.py
  > has our config for the dataset like source lang, target lang, epoch size,learning rate, where to store the files etc.
- src/dataset.py
  > builds on top of the dataset and adds out SOS and EOS tokens and the padding tokens ,also adds [causal mask](https://ai.stackexchange.com/questions/42116/transformer-decoder-causal-masking-during-inference)
- ./ruff.toml
  > python formatting rules for this project.
