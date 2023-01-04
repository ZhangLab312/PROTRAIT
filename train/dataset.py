import pandas as pd
from torch.utils.data import DataLoader

from public.loader import Seq2Ab, SampleLoader


class MyDataset(Seq2Ab):
    def __init__(self, mode):
        super(MyDataset, self).__init__()

        self.sequence = pd.read_csv("../data/" + mode + "_seq.csv",
                                    sep=',',
                                    header=None)[1].tolist()

        self.label = pd.read_csv("../data/" + mode + "_labels.csv",
                                 sep=',',
                                 header=None).values
        super().onehot_process_seq()
        super().process_label()

    def __getitem__(self, item):
        cur_sequence = self.sequences[item]
        cur_label = self.labels[item]
        return cur_sequence, cur_label

    def __len__(self):
        return len(self.sequence)


class MyDataLoader(SampleLoader):
    def __init__(self, mode="learn", batch_size=None):
        super(MyDataLoader, self).__init__()

        shuffle = True if mode == "learn" else False
        drop_last = True if mode == 'learn' else False

        self.loader = DataLoader(dataset=MyDataset(mode),
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=0,
                                 drop_last=drop_last)

    def __call__(self, *args, **kwargs):
        return self.loader
