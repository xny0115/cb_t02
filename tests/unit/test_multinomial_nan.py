import torch
from src.model.transformer import Seq2SeqTransformer


class NanModel(Seq2SeqTransformer):
    def __init__(self):
        super().__init__(vocab_size=5)

    def forward(self, src, tgt):
        return torch.full((tgt.size(0), src.size(1), 5), float('nan'))


def test_multinomial_nan():
    model = NanModel()
    src = torch.tensor([[0]])
    out = model.generate(src, max_new_tokens=2)
    ids = out.view(-1).tolist()[1:]
    assert ids != []
