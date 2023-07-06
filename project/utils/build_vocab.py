import os
import torchtext
import sentencepiece as spm
from torchtext.data.functional import sentencepiece_numericalizer

TRAIN_FILE = "../data/rdf/train.csv"
EVAL_FILE = "../data/rdf/validation.csv"
TEST_FILE = "../data/rdf/test.csv"
OUTPUT_DIR = "../vocab/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sp_model = torchtext.data.functional.generate_sp_model(CSV_FILE, vocab_size=25000, model_type="bpe", model_prefix=OUTPUT_DIR + "sentencepiece")

# spm.SentencePieceTrainer.Train(
#     '--input={},{},{} --model_prefix={} --vocab_size={} --model_type={} --user_defined_symbols={},{}'.format(
#         TRAIN_FILE, EVAL_FILE, TEST_FILE, OUTPUT_DIR + "sentencepiece", 20000, "bpe", "<pad>",
#         "<tsp>"))

sp_model = torchtext.data.functional.load_sp_model(OUTPUT_DIR + "sentencepiece.model")
vocab = [[sp_model.IdToPiece(id), id] for id in range(sp_model.GetPieceSize())]
print(vocab[:10])

# generator = sentencepiece_numericalizer(sp_model)
# print(list(generator(["a | b || c"])))
