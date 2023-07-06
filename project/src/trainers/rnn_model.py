import os
import numpy as np
import torch
import torch.nn as nn

from src.trainers.layer.EncoderDecoder import EncoderDecoder, Generator
from src.trainers.layer.Decoder import Decoder
from src.trainers.layer.DocumentEncoder import TextEncoder
from src.trainers.layer.QueryEncoder import QueryEncoder
from src.trainers.layer.Attention import BahdanauAttention
from src.trainers.layer.Loss import SimpleLossCompute
from src.dataloader.load_data import load_data
import math, copy, time
from tqdm import tqdm
import matplotlib.pyplot as plt
from rouge import Rouge

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rouge = Rouge()


def make_model(args):
    attention = BahdanauAttention(args.hidden_size)
    model = EncoderDecoder(
        TextEncoder(args.embed_size, args.hidden_size, num_layers=args.num_layers,
                    dropout=args.dropout),
        QueryEncoder(args.embed_size, args.hidden_size, num_layers=args.num_layers,
                     dropout=args.dropout),
        Decoder(args.embed_size, args.hidden_size, attention, num_layers=args.num_layers,
                dropout=args.dropout),
        nn.Embedding(args.vocab_size, args.embed_size),
        nn.Embedding(args.vocab_size, args.embed_size),
        nn.Embedding(args.vocab_size, args.embed_size),
        Generator(args.hidden_size, args.vocab_size))
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model.cuda() if args.use_cuda else model, criterion, optim


def get_loader(args):
    train_loader, validation_loader, test_loader = load_data(args)
    return train_loader, validation_loader, test_loader


def save_checkpoint(model, optimizer, ind, epoch, loss, args):
    checkpoint = {
        'epoch': epoch,
        'time_step': ind,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(checkpoint, args.model_dir + "checkpoint.pt")
    print()
    print("saved the checkpoint...")


def load_checkpoint(model, optimizer, args):
    """
    load latest checkpoint
    """
    if os.path.exists(args.model_dir + "checkpoint.pt"):
        checkpoint = torch.load(args.model_dir + "checkpoint.pt")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(
            checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print()
        print("loading model from checkpoint, epoch: {} loss: {}".format(epoch, loss))

    return model, optimizer


def run_epoch(data_iter, model, loss_compute, optim, epoch, args, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(tqdm(data_iter), 1):
        text, query, summary, summary_y, text_len, query_len, summary_len, text_mask, query_mask, summary_mask, ntokens = batch
        out, _, pre_output, text_total_loss, query_total_loss = model.forward(text, query, summary, text_mask,
                                                                              query_mask,
                                                                              summary_mask, text_len,
                                                                              query_len,
                                                                              summary_len)
        loss = loss_compute(pre_output, summary_y, text.size(0), text_total_loss, query_total_loss)
        total_loss += loss
        total_tokens += ntokens
        print_tokens += ntokens

        if i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / text.size(0), print_tokens / elapsed))
            start = time.time()
            print_tokens = 0
        if model.training and i % int(10 * print_every) == 0:
            save_checkpoint(model, optim, i, epoch, loss / text.size(0), args)

    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model, text, text_mask, text_lengths, query, query_mask, query_lengths,
                  max_len=100, sos_index=1,
                  eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        text_encoder_hidden, text_encoder_final = model.text_encode(text, text_mask, text_lengths)
        query_encoder_hidden, query_encoder_final = model.query_encode(query, query_mask,
                                                                       query_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(text)
        summary_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output,text_total_loss, query_total_loss = model.decode(text_encoder_hidden, text_encoder_final,
                                                   text_mask, prev_y, summary_mask,
                                                   query_encoder_hidden, query_encoder_final,
                                                   query_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(text).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, sp_model):
    x = [sp_model.IdToPiece(i).replace("▁", " ") for i in x]
    return [str(t) for t in x]


def print_examples(example_iter, model, spm, n=2, max_len=100):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    text_eos_index = 2
    summary_sos_index = 1
    summary_eos_index = 2
    rogue_1 = 0
    rogue_2 = 0
    rogue_l = 0
    for i, batch in enumerate(example_iter):
        text, query, summary, summary_y, text_len, query_len, summary_len, text_mask, query_mask, summary_mask, ntokens = batch
        _text = text.cpu().numpy()[0, :]
        _query = query.cpu().numpy()[0, :]
        _summary = summary_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        _text = _text[:-1] if _text[-1] == text_eos_index else _text
        _query = _query[:-1] if _query[-1] == text_eos_index else _query
        _summary = _summary[:-1] if _summary[-1] == summary_eos_index else _summary

        result, _ = greedy_decode(model,
                                  text,
                                  text_mask,
                                  text_len,
                                  query,
                                  query_mask,
                                  query_len,
                                  max_len=max_len,
                                  sos_index=summary_sos_index,
                                  eos_index=summary_eos_index)
        hypothesis = "".join(lookup_words(result, sp_model=spm))
        reference = "".join(lookup_words(_summary, sp_model=spm))
        scores = rouge.get_scores(hypothesis, reference)
        rogue_1 += scores[0]['rouge-1']["p"]
        rogue_2 += scores[0]['rouge-2']["p"]
        rogue_l += scores[0]['rouge-l']["p"]
        print("Example #%d" % (i + 1))
        print("text : ", "".join(lookup_words(_text, sp_model=spm)))
        print("query : ", "".join(lookup_words(_query, sp_model=spm)))
        print("summary : ", reference)
        print("Pred: ", hypothesis)
        print()

        count += 1
        if count == n:
            break

    print("=" * 10)
    print("rogue1: ", rogue_1 / n)
    print("rogue2: ", rogue_2 / n)
    print("roguel: ", rogue_l / n)
    print()


def run(model, criterion, optim, train_loader, validation_loader, test_loader, args):
    """Train the simple copy task."""

    if args.use_cuda:
        model.cuda()

    for epoch in range(args.epochs):
        print("Epoch %d" % epoch)

        # train
        model.train()
        if args.do_train:
            # data = data_gen(num_words=num_words, batch_size=args.batch_size, num_batches=100)
            run_epoch(train_loader, model, SimpleLossCompute(model.generator, criterion, optim),
                      optim,
                      epoch, args)

        # evaluate
        model.eval()
        with torch.no_grad():
            if args.do_test:
                print_examples(validation_loader, model, n=args.sample_n, max_len=args.max_length,
                               spm=args.spm)
            if args.do_eval:
                perplexity = run_epoch(validation_loader, model,
                                       SimpleLossCompute(model.generator, criterion, None), None,
                                       epoch,
                                       args)
                print("\nEvaluation perplexity: %f" % perplexity)


def start(args):
    print("running on {}".format("cuda" if args.use_cuda else "cpu"))
    model, criterion, optim = make_model(args)
    if args.resume:
        model, optim = load_checkpoint(model, optim, args)
    train_loader, validation_loader, test_loader = get_loader(args)
    run(model, criterion, optim, train_loader, validation_loader, test_loader, args)
