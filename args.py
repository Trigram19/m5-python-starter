import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--d_model', type=int, default=0, help='d_model')
parser.add_argument('--d_head', type=int, default=2, help='head')
parser.add_argument('--d_inner', type=bool, default=True, help='inner layers')

parser.add_argument('--n_token', type=str, default='roberta-base', help='number of tokens')
parser.add_argument('--n_layer', type=str, default='gru', help='number of hidden layers')
parser.add_argument('--n_head', type=int, default=2, help='num attention heads')

parser.add_argument('--dropout', type=int, default=1024, help='dropout')
parser.add_argument('--dropatt', type=int, default=0.5, help='dropatt')

parser.add_argument('--attention_dropout_prob', type=int, default=1024, help='attention_dropout_prob')
parser.add_argument('--output_dropout_prob', type=int, default=0.5, help='output_dropout_prob')


args = parser.parse_args()
args = vars(args)
