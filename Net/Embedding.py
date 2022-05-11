import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class Entity_Aware_Embedding(nn.Module):
    def __init__(self, opt):
        super(Entity_Aware_Embedding, self).__init__()
        self.word_embedding = BertModel.from_pretrained(opt.bert_path)
        self.entity_embedding = BertModel.from_pretrained(opt.bert_path)

        for param in self.word_embedding.parameters():
            param.requires_grad = True
        for param in self.entity_embedding.parameters():
            param.requires_grad = True
        self.pos1_embedding = nn.Embedding(opt.pos_size + 1, opt.pos_dim)
        self.pos2_embedding = nn.Embedding(opt.pos_size + 1, opt.pos_dim)
        self.pos3_embedding = nn.Embedding(opt.pos_size + 1, opt.pos_dim)
        self.pos4_embedding = nn.Embedding(opt.pos_size + 1, opt.pos_dim)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.pos3_embedding.weight)
        nn.init.xavier_uniform_(self.pos4_embedding.weight)

    def forward(self, X, X_Pos1, X_Pos2, X_Pos3, X_Pos4, X_Ent1, X_Ent2):
        X, _ = self.word_embedding(X, attention_mask=None, output_all_encoded_layers=False)
        Xp = self.word_pos_embedding(X, X_Pos1, X_Pos2, X_Pos3, X_Pos4)
        Xe = self.word_ent_embedding(X, X_Ent1, X_Ent2)

        return Xp, Xe

    def word_pos_embedding(self, X, X_Pos1, X_Pos2, X_Pos3, X_Pos4):
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        X_Pos3 = self.pos3_embedding(X_Pos3)
        X_Pos4 = self.pos4_embedding(X_Pos4)
        return torch.cat([X, X_Pos1, X_Pos2, X_Pos3, X_Pos4], -1)

    def word_ent_embedding(self, X, X_Ent1, X_Ent2):
        X_Ent1, _ = self.entity_embedding(X_Ent1, attention_mask=None, output_all_encoded_layers=False)
        X_Ent2, _ = self.entity_embedding(X_Ent2, attention_mask=None, output_all_encoded_layers=False)
        X_Ent1 = X_Ent1.mean(dim=1).unsqueeze(1).expand(X.shape)
        X_Ent2 = X_Ent2.mean(dim=1).unsqueeze(1).expand(X.shape)
        out = torch.cat([X, X_Ent1, X_Ent2], -1)
        return out
