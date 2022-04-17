import argparse
from transformers.models import t5

parser = argparse.ArgumentParser(description='Model parameters')

parser.add_argument('--random_seed', type=int, default=1022, help='Choose random_seed')
parser.add_argument('--model', default='Bart',help='Sevral models are available: Bart | Bert | T5')
parser.add_argument('--epochs', type=int, default=40, help='Set training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--dropout', type=float, default=0.4,help='Set dropout rate')
parser.add_argument('--lr', type=float, default=5e-3, help='Set learning rate')
parser.add_argument('--optim', default='Adam', help='Choose optimizer')
parser.add_argument('--earlystop', type=bool, default=False, help='Apply EarlyStop')
parser.add_argument('--warmup', type=bool, default=False, help='Apply WarmUp')
parser.add_argument('--lrscheduler', type=bool, default=False, help='Apply LRScheduler')
parser.add_argument('--status', default='train',help='train, test, or infer')
parser.add_argument('--device', default='cuda:0',help='Device')


args = parser.parse_args()
