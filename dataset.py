from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        instance_file = './project-data/' + mode + '.data.txt'
        label_file = './project-data/' + mode + '.label.txt'

        f = open(instance_file)
        self.instance_lines = f.readlines()
        f.close()

        f = open(label_file)
        self.label_lines = f.readlines()
        f.close()

        assert len(self.instance_lines) == len(self.label_lines), "Inconsistant instances and labels"

    def __getitem__(self, index):
        return self.instance_lines[index], self.label_lines[index]

    def __len__(self):
        return len(self.instance_lines)


if __name__ == '__main__':

    test_set = MyDataset(mode = 'dev')

    files, label = test_set[0]

    temp = files.strip('\n').split(',')

    for item in temp:
        f = open('./project-data/tweet-objects' + temp + '.json')