import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=int, default=8e-3)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--Lambda", type=int, default=1.75)
parser.add_argument("--train_set_directory", type=str, default='./train_set/')
parser.add_argument("--save_directory", type=str, default='./save/')
parser.add_argument("--Backbone_model", type=str, default='VGG16')
parser.add_argument("--use_multiprocessing", type=bool, default=True)
parser.add_argument("--show_ModelSummary", type=bool, default=False)


cfg = parser.parse_args()
