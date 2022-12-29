import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def one_hot(seq):
    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    code = np.empty(shape=(len(seq), 4), dtype=np.int8)
    for location, base in enumerate(seq, start=0):
        code[location] = base_map[base]

    return code


class Seq2Ab(Dataset):
    def __init__(self):

        super(Seq2Ab, self).__init__()
        self.inputFile = None
        self.labelFile = None
        self.sequence = None
        self.labels = None
        self.sequences = None
        self.label = None
        self.label_value = "binary"

    def onehot_process_seq(self):
        """
        sequence type: list
        :return:
        """
        seq_num = len(self.sequence)
        seq_len = 1344

        self.sequences = np.empty(shape=(seq_num, seq_len, 4), dtype=np.int8)
        for i in tqdm(range(seq_num), postfix="data loading"):
            self.sequences[i] = one_hot(self.sequence[i])
        self.sequences = np.transpose(self.sequences, [0, 2, 1])

    def process_label(self):
        self.labels = np.array(self.label, dtype=np.int8)
        if self.label_value == "binary":
            self.labels[self.labels > 1] = 1

    def __getitem__(self, index):
        cur_sequence = self.sequences[index]
        cur_label = self.labels[index]
        return cur_sequence, cur_label

    def __len__(self):
        sample_num = len(self.sequence)
        return sample_num


class SampleLoader:
    def __init__(self):
        self.mode = None
        self.loader = None

    def __call__(self):
        return self.loader