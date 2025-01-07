import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # mapping the unknown
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # unknown, padding, start of sentence ,end of sentence
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )

        # train the tokenizer
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_all_sentences(ds, lang):
    # each is a pair of sentences, one in english , one in italian
    for item in ds:
        yield item["translation"][lang]


def get_dataset(config):
    ds_raw = load_dataset(
        "opus_books",
        f"{config["lang_src"]}-{config["lang_tgt"]}",
        split="train",
    )

    # build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config=["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.random_split(
        ds_raw, [train_ds_size, val_ds_size]
    )
    
