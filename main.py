import argparse
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import FixedScheduler, WarmupLinearScheduler, save
from metrics import calc_accuracy_score, calc_f1_score
from dataset import MyDataset, Collator

parser = argparse.ArgumentParser(description='Model parameters')

parser.add_argument('--random_seed', type=int, default=1022, help='Choose random_seed')
parser.add_argument('--model', default='bart',help='Sevral models are available: Bart | Bert | T5')
parser.add_argument('--model_path', default='./models/bart-base/checkpoint/bart-step-1600', help="use local model weights if specified")
parser.add_argument('--save_path', default='./models/bart-base', help="where to save the model")
parser.add_argument('--model_size', type=str, default='small', help='PLM model size: small | base | large')
parser.add_argument('--total_steps', type=int, default=2000, help='Set training steps')
parser.add_argument('--eval_steps', type=int, default=400, help='Set evaluation steps')
parser.add_argument('--batch_size', type=int, default=4, help='Set batch size')
parser.add_argument('--dropout', type=float, default=0.4,help='Set dropout rate')
parser.add_argument('--lr', type=float, default=5e-3, help='Set learning rate')
parser.add_argument('--optim', default='Adam', help='Choose optimizer')
parser.add_argument('--warmup_steps', type=int, default=1000, help='WarmUp Steps')
parser.add_argument('--lrscheduler', action='store_true', help='Apply LRScheduler')
parser.add_argument('--mode', default='test',help='train, test, or inference')
parser.add_argument('--device', default='cuda:0',help='Device')

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # load models
    if args.model == 'bart':
        if args.model_path == "none":
            model = transformers.BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels = 2)
            tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
        else:
            model = transformers.BartForSequenceClassification.from_pretrained(args.model_path, num_labels = 2)
            tokenizer = transformers.BartTokenizer.from_pretrained(args.model_path)
        model.to(args.device)

    elif args.model == 'roberta':
        if args.model_path == "none":
            model = transformers.RobertaForSequenceClassification.from_pretrained(num_labels = 2)
            tokenizer = transformers.RobertaTokenizer.from_pretrained(num_labels = 2)
        else:
            model = transformers.RobertaForSequenceClassification.from_pretrained(args.model_path, num_labels = 2)
            tokenizer = transformers.RobertaTokenizer.from_pretrained(args.model_path, num_labels = 2)
        model.to(args.device)

    elif args.model == 'bert':
        if args.model_path == "none":
            model = transformers.BertForSequenceClassification.from_pretrained(num_labels = 2)
            tokenizer = transformers.BertTokenizer.from_pretrained(num_labels = 2)
        else:
            model = transformers.BertForSequenceClassification.from_pretrained(args.model_path, num_labels = 2)
            tokenizer = transformers.BertTokenizer.from_pretrained(args.model_path, num_labels = 2)
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
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, collate_fn=collator)
        step, epoch = 0, 0
        loss, curr_loss = 0.0, 0.0
        best_f1 = 0
        model.train()
        model.zero_grad()
        while step < args.total_steps:
            epoch += 1
            with tqdm(train_loader) as t:
                t.set_description("Epochs "+ str(epoch))
                for i, batch in enumerate(t):
                    step += 1
                    (texts, labels) = batch
                    # print(texts, labels)
                    train_loss = model(**texts.to(args.device), labels=labels.to(args.device)).loss

                    train_loss.backward()

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    curr_loss += train_loss.item()

                    if step % args.eval_steps == 0:
                        model.eval()
                        # evaluation
                        dev_sampler = SequentialSampler(dev_set)
                        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, sampler=dev_sampler, num_workers=0, collate_fn=collator)
                        all_pred_labels = []
                        all_ture_labels = []
                        with tqdm(dev_loader) as t:
                            t.set_description("Dev")
                            for i, batch in enumerate(t):
                                (texts, labels) = batch
                                all_ture_labels += labels
                                output = model(**texts.to(args.device), labels=labels.to(args.device)).logits
                                
                                all_pred_labels += [item.argmax().item() for item in output]
                        # metrics here
                        acc = calc_accuracy_score(all_pred_labels, all_ture_labels)
                        f1, precision, recall = calc_f1_score(all_pred_labels, all_ture_labels)
                        print("Step {}: Acc = {:2f}; F1 = {:2f}, P = {:2f}; R = {:2f}\n".format(step, acc, f1, precision, recall))
                        # save best
                        if f1 > best_f1:
                            save(model, tokenizer, args.save_path, args.model+'-step-'+str(step))
                            best_f1 = f1
                        model.train()

    # testing
    elif args.mode == 'test':
        model.eval()
        test_sampler = RandomSampler(test_set)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=0, collate_fn=collator)

        labels = []
        with tqdm(test_loader) as t:
            t.set_description('Test')
            for i, batch in enumerate(t):
                text = batch
                output = model(**text.to(args.device)).logits

                labels += [item.argmax().item() for item in output]
        
        with open('./project-data/test_pred.label.csv', 'w+') as f:
            f.write('Id,Predicted')
            for i in range(len(labels)):
                f.write(str(i) + ',' + str(labels[i]))


    # inference
    elif args.mode == 'inference':
        model.eval()
        text = input("Please input the text:")
        text = tokenizer(text, return_tensors='pt')
        output = model(**text.to(args.device))

    else:
        raise RuntimeError("[Error] Mode not in the scope!")