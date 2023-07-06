""" File to hold arguments """
import argparse

# data arguments
parser = argparse.ArgumentParser(description="Main Arguments")

# parser.add_argument(
#     '--epochs', type=int, default=3,
#     required=True, help='Number of epochs (deprecated)')
# parser.add_argument(
#     '--batch_size', type=int, required=True, help='Batch size')


def get_args():
    args = parser.parse_args()
    return args
