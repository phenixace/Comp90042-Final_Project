import argparse
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import FixedScheduler, WarmupLinearScheduler, save
from dataset import MyDataset, Collator

parser = argparse.ArgumentParser(description='Model parameters')

parser.add_argument('--random_seed', type=int, default=1022, help='Choose random_seed')
parser.add_argument('--model', default='none',help='Sevral models are available: Bart | Bert | T5')
parser.add_argument('--model_path', default='none', help="use local model weights if specified")
parser.add_argument('--model_size', type=str, default='small', help='PLM model size: small | base | large')
parser.add_argument('--total_steps', type=int, default=10000, help='Set training steps')
parser.add_argument('--eval_steps', type=int, default=1000, help='Set evaluation steps')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--dropout', type=float, default=0.4,help='Set dropout rate')
parser.add_argument('--lr', type=float, default=5e-3, help='Set learning rate')
parser.add_argument('--optim', default='Adam', help='Choose optimizer')
parser.add_argument('--warmup_steps', type=int, default=1000, help='WarmUp Steps')
parser.add_argument('--lrscheduler', action='store_true', help='Apply LRScheduler')
parser.add_argument('--mode', default='train',help='train, test, or inference')
parser.add_argument('--device', default='cuda:0',help='Device')


args = parser.parse_args()

torch.manual_seed(args.random_seed)

# load models
if args.model == 'bart':
    if args.model_path == "none":
        model = transformers.BartForSequenceClassification.from_pretrained('facebook/bart-small', num_labels = 1)
        tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-small')
    else:
        model = transformers.BartForSequenceClassification.from_pretrained(args.model_path, num_labels = 1)
        tokenizer = transformers.BartTokenizer.from_pretrained(args.model_path)
    model.to(args.device)

elif args.model == 'roberta':
    if args.model_path == "none":
        model = transformers.RobertaForSequenceClassification.from_pretrained(num_labels = 1)
        tokenizer = transformers.RobertaTokenizer.from_pretrained(num_labels = 1)
    model.to(args.device)

elif args.model == 'bert':
    if args.model_path == "none":
        model = transformers.BertForSequenceClassification.from_pretrained(num_labels = 1)
        tokenizer = transformers.BertTokenizer.from_pretrained(num_labels = 1)
    model.to(args.device)

else:
    raise RuntimeError("[Error] Model not in the scope!")

# load data
train_set = MyDataset('train')
dev_set = MyDataset('dev')
test_set = MyDataset('test')

collator = Collator(tokenizer, max_length=512)

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
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=10, collate_fn=collator)
    step, epoch = 0, 0
    loss, curr_loss = 0.0, 0.0
    model.train()
    model.zero_grad()
    while step < args.total_steps:
        epoch += 1
        with tqdm(train_loader) as t:
            t.set_description("Epochs: "+ str(epoch))
            for i, batch in enumerate(t):
                step += 1
                (text, labels) = batch

                train_loss = model(input_ids=text.to(args.device), labels=labels.to(args.device)).loss

                train_loss.backward()

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                curr_loss += train_loss.item()

                if step % args.eval_steps == 0:
                    model.eval()
                    # evaluation
                    dev_sampler = SequentialSampler(dev_set)
                    dev_loader = DataLoader(dev_set, bacth_size=args.bacth_size, sampler=dev_sampler, num_workers=10, collate_fn=collator)
                    with tqdm(dev_loader) as t:
                        t.set_description("Dev: ")
                        for i, batch in enumerate(t):
                            (text, label) = batch
                            model(input_ids=text.to(args.device), labels=labels.to(args.device))
                            # metrics here
                    # save checkpoint
                    save(model)
                    model.train()

# testing
elif args.mode == 'test':
    model.eval()
    test_sampler = RandomSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=10, collate_fn=collator)

    for i, batch in enumerate(tqdm(test_loader)):
        text = batch
        model(input_ids=text.to(args.device))

# inference
elif args.mode == 'inference':
    model.eval()

else:
    raise RuntimeError("[Error] Mode not in the scope!")