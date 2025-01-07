import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


from dataset import BillingualDataset, causal_mask
from model import build_transformer

from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path, get_config


from pathlib import Path
import warnings


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
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train",
    )

    # build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size]
    )

    # create 2 dataset -> one for training , one for validation
    train_ds = BillingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BillingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item["translation"][config("lang_src")]
        ).ids
        tgt_ids = tokenizer_src.encode(
            item["translation"][config("lang_tgt")]
        ).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    # batch_size as 1 , to process each sentence one by one
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    # device on which we put all the tensors
    # ie check if cuda present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    (
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    ) = get_dataset(config)

    model = get_model(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)
    # tensorboard allows to visualize the chart repr of loss
    writer = SummaryWriter(config["experiment_name"])

    # using adam as an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config("lr"), eps=1e-9)

    # resume the training if crashes
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # using cross entropy loss
    # label_smoothing so as to distribute bias and improve accuracy
    # of the model
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing epoch {epoch:02d}"
        )
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (0,seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (0,seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (0,1,1,seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (0,1,seq_len,seq_len)

            # run the tensors through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (0,seq_len,d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (0,seq_len,d_model)
            # need to map into back into vocab, so need a projection
            proj_output = model.project(
                decoder_output
            )  # (0,seq_len,tgt_vocab_size)

            # extract label from the batch
            label = batch["label"].to(device)  # (0,seq_len)

            # compute the loss into projection output view
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )  # (0,seq_len,tgt_vocab_size) -> (0*seq_len,tgt-vocab_size)
            # update the progress bar with the loss that we have calcualted
            batch_iterator.set_postfix(f"Loss: {loss.item():0.2f}")
            # logging into tensorboard
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            ## backpropogate the loss
            loss.backward()

            # update the weights
            optimizer.stop()
            optimizer.zero_grad()

            global_step += 1

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # save the state of the optimizer also
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
