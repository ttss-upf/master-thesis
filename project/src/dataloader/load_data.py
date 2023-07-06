from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dataloader.cnn_dataset import  CNNDataset

BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
SEP_ID = 4

def collate_pad(batch):
    (text, query, summary) = zip(*batch)
    text_pad = pad_sequence(text, batch_first=True, padding_value=PAD_ID)
    query_pad = pad_sequence(query, batch_first=True, padding_value=PAD_ID)
    summary_pad = pad_sequence(summary, batch_first=True, padding_value=PAD_ID)

    text_lens = [text_pad.size(1)] * text_pad.size(0)
    query_lens = [query_pad.size(1)] * query_pad.size(0)
    summary_lens = [summary_pad.size(1)] * summary_pad.size(0)

    summary_pad = summary_pad[:, :-1]
    summary_pad_y = summary_pad[:, 1:]
    text_mask = (text_pad != PAD_ID).unsqueeze(-2)
    query_mask = (query_pad != PAD_ID).unsqueeze(-2)
    summary_mask = (summary_pad_y != PAD_ID)

    ntokens = (summary_pad_y != PAD_ID).data.sum().item()

    return text_pad, query_pad, summary_pad, summary_pad_y, text_lens, query_lens, summary_lens, text_mask, query_mask, summary_mask, ntokens


def load_data(args):
    train_set = CNNDataset(args.data_dir + args.train_dir, args.spm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)

    validation_set = CNNDataset(args.data_dir + args.validation_dir, args.spm)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True, collate_fn=collate_pad)

    test_set = CNNDataset(args.data_dir + args.test_dir, args.spm)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_pad)

    return train_loader, validation_loader, test_loader

# from utils.arguments import get_args
# import torchtext
#
# args = get_args()
#
# args.train_dir = "train2.csv"
# args.test_dir = "test.csv"
# args.validation_dir = "validation2.csv"
# args.vocab_dir = "../../vocab/"
# args.model_name = "sentencepiece.model"
# args.data_dir = "../../data/cnn/"
# args.model_dir = "./models/"
#
# # args.train_dir = "train.csv"
# # args.test_dir = "test.csv"
# # args.validation_dir = "validation.csv"
# # args.vocab_dir = "./drive/MyDrive/"
# # args.model_name = "sentencepiece.model"
# # args.data_dir = "./drive/MyDrive/"
# # args.model_dir = "./drive/MyDrive/models/"
#
#
# args.vocab_size = 20000
# args.embed_size = 256
# args.num_layers = 1
# args.hidden_size = 512
#
# args.dropout = 0.1
# args.lr = 0.0003
# args.epochs = 10
# args.batch_size = 4
# args.max_length = 100
# args.resume = False
# args.spm = torchtext.data.functional.load_sp_model(
#     args.vocab_dir + "sentencepiece.model")
#
#
# train_loader, validation_loader = load_data(args)
# for i,batch in enumerate(train_loader):
#     text, query, summary, summary_y, text_len, query_len, summary_len, text_mask, query_mask, summary_mask, ntokens = batch
