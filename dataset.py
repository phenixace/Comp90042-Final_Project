from torch.utils.data import Dataset
import json
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
            return text, label
        else:
            return text

    def __len__(self):
        return len(self.instance_lines)


class Collator(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pass

if __name__ == '__main__':

    test_set = MyDataset(mode = 'test')

    text = test_set[1]

    print(text)
