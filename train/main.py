import torch

from public.model import Informer
from public.train import Builder
from train.dataset import MyDataLoader


class MyBuilder(Builder):

    def __init__(self, model, model_name):
        """
        :param model:
        :param model_name:
        """
        super().__init__(model, model_name)

    def learn(self, learn_loader, val_loader):
        super().learn(learn_loader, val_loader)

    def inference(self, infer_loader, measure=True):
        self.model.load_state_dict(torch.load(self.cwd + "/pth/" + self.model_name + ".pth"))
        super().inference(infer_loader, measure)

    def __call__(self, mode):
        if mode == "all" or mode == "learn":
            self.learn_loader = MyDataLoader(mode="learn", batch_size=self.batch_size).__call__()
            self.val_loader = MyDataLoader(mode="val", batch_size=self.batch_size).__call__()
        if mode == "all" or mode == "infer":
            self.infer_loader = MyDataLoader(mode="infer", batch_size=self.batch_size).__call__()
        super().__call__(mode)


if __name__ == "__main__":
    model = Informer(seq_len=1344)
    builder = MyBuilder(model, model_name="protrait")
    builder.batch_size = 1
    builder.__call__(mode="all")
