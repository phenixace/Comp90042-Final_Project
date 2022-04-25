from torch.utils.data import Dataset
import json
import torch
from utils import clean_text

class MyDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.sep = ' '
        instance_file = './project-data/' + mode + '.data.txt'   
        rm_file = './project-data/logs.txt' 

        with open(rm_file, 'r') as f:
            temp = f.readlines()
        
        with open(instance_file, 'r') as f:
            instance_lines = f.readlines()

        rm_instances = []
        for item in temp:
            rm_instances += item.strip('\n').split(',')

        self.instances = []
        not_found = []
        for i in range(0, len(instance_lines)):
            temp = instance_lines[i].strip('\n').split(',')
            cur = []
            for id in temp:
                if self.mode != 'test':
                    if id not in rm_instances:
                        cur.append(id)
                else:
                    cur.append(id)
            if len(cur) != 0:
                self.instances.append(cur)
            else:
                not_found.append(i)

        
        if self.mode != 'test':
            label_file = './project-data/' + mode + '.label.txt'
            self.labels = []
            with open(label_file) as f:
                label_lines = f.readlines()

            for i in range(0, len(label_lines)):
                if i not in not_found:
                    self.labels.append(label_lines[i])

            assert len(self.instances) == len(self.labels), "Inconsistant number between instances and labels"

    def __getitem__(self, index):
        temp = self.instances[index]
        text = ""
        for item in temp:
            f = open('./project-data/' +self.mode+ '-tweet-objects/' + item + '.json', 'r', encoding='utf-8')
            content = json.load(f)
            text += clean_text(content['text']).strip() + self.sep
            f.close()
        text = text.strip()
        if self.mode != 'test':       
            if self.labels[index].strip('\n') == "rumour": 
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
        return len(self.instances)

class Collator(object):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        text = [item['text'] for item in batch]
        text = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_length if self.max_length > 0 else None,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True if self.max_length > 0 else False,
        )

        if 'label' in batch[0]:
            label = torch.tensor([item['label'] for item in batch])
            return (text, label)   
        else:
            return text

class Dataset4SKEP(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.sep = ' '
        instance_file = './project-data/' + mode + '.data.txt'   
        rm_file = './project-data/logs.txt' 

        with open(rm_file, 'r') as f:
            temp = f.readlines()
        
        with open(instance_file, 'r') as f:
            instance_lines = f.readlines()

        rm_instances = []
        for item in temp:
            rm_instances += item.strip('\n').split(',')

        instances = []
        not_found = []
        for i in range(0, len(instance_lines)):
            temp = instance_lines[i].strip('\n').split(',')
            cur = []
            for id in temp:
                if self.mode != 'test':
                    if id not in rm_instances:
                        cur.append(id)
                else:
                    cur.append(id)
            if len(cur) != 0:
                instances.append(cur)
            else:
                not_found.append(i)

        
        if self.mode != 'test':
            label_file = './project-data/' + mode + '.label.txt'
            labels = []
            with open(label_file) as f:
                label_lines = f.readlines()

            for i in range(0, len(label_lines)):
                if i not in not_found:
                    labels.append(label_lines[i])

            assert len(instances) == len(labels), "Inconsistant number between instances and labels"
        
        self.data = []
        for i in range(0, len(instances)):
            temp = instances[i]
            text = ""
            for item in temp:
                f = open('./project-data/' +self.mode+ '-tweet-objects/' + item + '.json', 'r', encoding='utf-8')
                content = json.load(f)
                text += clean_text(content['text']).strip() + self.sep
                f.close()
            text = text.strip()
            if self.mode != 'test':       
                if labels[i].strip('\n') == "rumour": 
                    label = 1
                else:
                    label = 0
                self.data.append({
                    'text': text, 
                    'label': label,
                    'qid':i
                })
            else:
                self.data.append({
                    'text': text,
                    'label': 0,
                    'qid':i
                })

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    test_set = MyDataset(mode='test')
    print(len(test_set))
