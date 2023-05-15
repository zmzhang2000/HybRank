import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Hybrank(nn.Module):
    def __init__(self, in_dim=2, embed_dim=64, depth=2, num_heads=8):
        super().__init__()
        self.cnn = nn.Conv2d(in_dim, embed_dim, 1)
        self.embed_dim = embed_dim

        config = BertConfig(vocab_size=1, hidden_size=embed_dim, num_hidden_layers=depth,
                            num_attention_heads=num_heads, intermediate_size=int(embed_dim * 4))
        self.col_transformer = BertModel(config=config)

        self.out_cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.kaiming_uniform_(self.out_cls_token)
        config = BertConfig(vocab_size=1, hidden_size=embed_dim, num_hidden_layers=1,
                            num_attention_heads=num_heads, intermediate_size=int(embed_dim * 4))
        self.out_transformer = BertModel(config=config)

        self.t = 0.07

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)

        B, Nrow, Ncol, D = x.shape
        assert D == self.embed_dim

        x = x.transpose(1, 2).reshape(-1, Nrow, self.embed_dim)
        x = self.col_transformer(inputs_embeds=x)[0]
        x = x.reshape(B, Ncol, Nrow, self.embed_dim).transpose(1, 2)

        x = x.reshape(-1, Ncol, self.embed_dim)
        x = torch.cat((self.out_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.out_transformer(inputs_embeds=x)[0]
        x = x[:, 0, :]
        x = x.reshape(B, Nrow, self.embed_dim)
        x = F.normalize(x, dim=-1)
        q, p = x[:, :1], x[:, 1:]
        out = (q @ p.transpose(1, 2)).squeeze(1) / self.t
        return out


def compute_loss(input, target):
    """
    :param input: Float tensor of size (B, L)
    :param target: Bool tensor of size (B, L)
    :return: loss: Scalar tensor
    """
    # remove samples without positive
    mask = target.sum(dim=-1).bool()
    input, target = input[mask], target[mask]

    target = target.float()
    log_softmax = input.log_softmax(dim=-1)
    loss = -((log_softmax * target).sum(-1) / target.sum(-1)).mean()
    return loss
