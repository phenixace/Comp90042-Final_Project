from torch.utils.data import Dataset
import json
import torch
from utils import clean_text

class MyDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.sep = ' </s> '
        instance_file = './project-data/' + mode + '.data.txt'      

        f = open(instance_file)
        self.instance_lines = f.readlines()
        f.close()

        if self.mode != 'test':
            label_file = './project-data/' + mode + '.label.txt'
            f = open(label_file)
            self.label_lines = f.readlines()
            f.close()

            assert len(self.instance_lines) == len(self.label_lines), "Inconsistant number between instances and labels"

    def __getitem__(self, index):
        temp = self.instance_lines[index].strip('\n').split(',')
        text = ""
        for item in temp:
            f = open('./project-data/tweet-objects/' + item + '.json')
            content = json.load(f)
            text += clean_text(content['text']).strip() + self.sep
        text = text.strip()
        if self.mode != 'test':       
            if self.label_lines[index].strip('\n') == "rumour": 
                label = 1
            else:
                label = 0
            return {
                'text': text, 
                'label': label
            }
        else:
            return {
                'text': text
            }

    def __len__(self):
        return len(self.instance_lines)


class Collator(object):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        text = [item['text'] for item in batch]
        text = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_length if self.max_length > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.max_length > 0 else False,
        )

        if 'label' in batch[0]:
            label = torch.tensor([item['label'] for item in batch])
            return (text, label)
        else:
            return text


if __name__ == '__main__':

    instance_file = './project-data/' + 'train' + '.data.txt'      

    f = open(instance_file)
    instance_lines = f.readlines()
    f.close()

    num = 0
    for line in instance_lines:
        num += len(line.split(','))

    print(num)