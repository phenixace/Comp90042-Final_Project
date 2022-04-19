import argparse
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils import FixedScheduler, WarmupLinearScheduler

parser = argparse.ArgumentParser(description='Model parameters')

parser.add_argument('--random_seed', type=int, default=1022, help='Choose random_seed')
parser.add_argument('--model', default='none',help='Sevral models are available: Bart | Bert | T5')
parser.add_argument('--model_size', type=str, default='small', help='PLM model size: small | base | large')
parser.add_argument('--total_steps', type=int, default=10000, help='Set training steps')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--dropout', type=float, default=0.4,help='Set dropout rate')
parser.add_argument('--lr', type=float, default=5e-3, help='Set learning rate')
parser.add_argument('--optim', default='Adam', help='Choose optimizer')
parser.add_argument('--warmup_steps', type=int, default=1000, help='WarmUp Steps')
parser.add_argument('--lrscheduler', action='store_true', help='Apply LRScheduler')
parser.add_argument('--status', default='train',help='train, test, or inference')
parser.add_argument('--device', default='cuda:0',help='Device')


args = parser.parse_args()

torch.manual_seed(args.random_seed)

# load models
if args.model == 'bart':
    model = transformers.BartForSequenceClassification.from_pretrained()
    tokenizer = transformers.BartTokenizer.from_pretrained()

elif args.model == 'roberta':
    model = transformers.RobertaForSequenceClassification.from_pretrained()
    tokenizer = transformers.RobertaTokenizer.from_pretrained()

elif args.model == 'bert':
    model = transformers.BertForSequenceClassification.from_pretrained()
    tokenizer = transformers.BertTokenizer.from_pretrained()

else:
    raise RuntimeError("[Error] Model not in the scope!")

# load data
data_loader = DataLoader()


# optimizer
if args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

# scheduler
if args.lrscheduler:
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=args.warmup_steps, scheduler_steps=args.total_steps, min_ratio=0., fixed_lr=args.fixed_lr)
else:
    scheduler = FixedScheduler(optimizer)

# training
if args.mode == 'train':
    step, epoch = 0, 0
    while step < args.total_steps:
        epoch += 1
        with tqdm(data_loader) as t:
            t.set_description("Epochs: "+ str(epoch))
            for instance, label in enumerate(t):
                pass

# testing
elif args.mode == 'test':
    pass

# inference
elif args.mode == 'inference':
    pass

else:
    raise RuntimeError("[Error] Mode not in the scope!")