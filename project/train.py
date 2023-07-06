from utils.arguments import get_args
from src.trainers.rnn_model import start
import torch
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
print("cuda is available: ", USE_CUDA)
import torchtext


args = get_args()

args.train_dir = "train.csv"
args.test_dir = "test.csv"
args.validation_dir = "test.csv"
args.data_dir = "./data/rdf/"
args.vocab_dir = "./vocab/"
args.model_dir = "./models/"

# args.train_dir = "train.csv"
# args.test_dir = "test.csv"
# args.validation_dir = "validation.csv"
# args.vocab_dir = "./drive/MyDrive/vocab/"
# args.data_dir = "./drive/MyDrive/cnn/"
# args.model_dir = "./drive/MyDrive/models/"


args.vocab_size = 20000
args.embed_size = 256
args.num_layers = 1
args.hidden_size = 512

args.dropout = 0.1
args.lr = 0.0003
args.epochs = 2
args.batch_size = 4
args.max_length = 512
args.use_cuda = USE_CUDA
args.spm = torchtext.data.functional.load_sp_model(
    args.vocab_dir + "sentencepiece.model")

args.do_train = True
args.do_eval = True
args.do_test = True
args.resume = False

args.sample_n = 5

start(args)