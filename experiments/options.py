import argparse
from datetime import datetime

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(self.parser)

    def initialize(self, parser):

        parser.add_argument('--dataset', type=str, default='MNIST', help='ratio of dataset A')

        parser.add_argument('--ratio_A', type=float, default=0.2, help='ratio of dataset A')
        parser.add_argument('--ratio_B', type=float, default=0.5, help='ratio of dataset B')

        parser.add_argument('--importance_sampling', type=int, default=1, help='importance sampling (1), batch weight (0)')

        parser.add_argument('--batch_size_A', type=int, default=256, help='batch size of A')
        parser.add_argument('--batch_size_B', type=int, default=64, help='batch size of B')
        parser.add_argument('--sampled_batch_size', type=int, default=64, help='sampled batch size')

        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

        parser.add_argument('--n_epochs', type=int, default=4, help='number of epochs')
        parser.add_argument('--max_steps', type=int, default=5000, help='number of training steps')

        parser.add_argument('--objective_function', type=int, default=4, help='the objective function')

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--experiment_name', type=str, default=datetime.now().strftime("%d%m%Y%H%M%S"))
        parser.add_argument('--CSV_name', type=str, default='output.csv')

        return parser
