import torch
import torch.nn as nn
from .Embedding import Entity_Aware_Embedding
from .Encoder import PCNN, SAN


class SeG(nn.Module):
    # def __init__(self, rel_num, lambda_pcnn=0.05, lambda_san=1.0, pos_dim=5, pos_len=100, hidden_size=230,
    #              dropout_rate=0.5):
    def __init__(self, opt):
        super(SeG, self).__init__()
        self.opt = opt
        self.embedding = Entity_Aware_Embedding(self.opt)  # Xp, Xe
        self.PCNN = PCNN(self.opt)
        self.SAN = SAN(self.opt)
        self.fc1 = nn.Linear(3 * opt.word_dim, 3 * opt.word_dim)
        self.fc2 = nn.Linear(3 * opt.word_dim, 3 * opt.hidden_size)
        self.classifer = nn.Linear(3 * opt.hidden_size, opt.rel_num)
        self.dropout = nn.Dropout(opt.drop_out)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.classifer.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.classifer.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2, X_Mask, X_Scope):
        # Embed
        Xp, Xe = self.embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        # Encode SAN与PCNN的entity-aware不共享参数
        S = self.PCNN(Xp, Xe, X_Mask)
        U = self.SAN(Xp, Xe)
        # # Combine
        X = self.selective_gate(S, U, X_Scope)
        print(X.size())
        X = self.dropout(X)
        # Classifier
        X = self.classifer(X)
        return X

    def selective_gate(self, S, U, X_Scope):
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))
        X = G * S
        print(X.size())
        B = []  # Bag Output
        for s in X_Scope:
            B.append(X[s[0]:s[1]])
        B = torch.stack(B)
        return B
