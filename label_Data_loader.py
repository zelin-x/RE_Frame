from codecs import open
import torch.utils.data as data
import torch
from config import DefaultConfig


class Label_Dataset(data.Dataset):

    def __init__(self, opt, input_path):
        super(Label_Dataset, self).__init__()
        self.data_dir = input_path  # data
        self.max_len = opt.max_len
        self.entity_max_len = opt.entity_max_len
        self.limit = opt.limit
        self.bert_vocab_path = opt.bert_vocab_path

        self.char2id = self.build_bert_vocab()
        self.sents = []
        self._preprocess()

    def build_bert_vocab(self):
        """Loads a vocabulary file into a dictionary."""
        vocab = {}
        index = 0
        with open(self.bert_vocab_path, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def _preprocess(self):
        # Load files
        print("Loading data file...")
        with open(self.data_dir, 'r', encoding='UTF-8') as f:
            lins = [_.strip() for _ in f]
            for index, lin in enumerate(lins):
                e1, e2, r, sent_str = lin.split('\t')

                e1_str, e1_pos = e1.split('&')
                e2_str, e2_pos = e2.split('&')

                # words
                sent_bert_index = []
                for _ in sent_str:
                    sent_bert_index.append(self.char2id.get(_, self.char2id['[UNK]']))

                # entities
                e1_ids = [self.char2id.get(_, self.char2id['[UNK]']) for _ in e1_str]
                e2_ids = [self.char2id.get(_, self.char2id['[UNK]']) for _ in e2_str]

                e1_ids = [self.char2id['[CLS]']] + e1_ids + [self.char2id['[SEP]']]
                e2_ids = [self.char2id['[CLS]']] + e2_ids + [self.char2id['[SEP]']]

                # entity_pad
                if len(e1_ids) > self.entity_max_len:
                    e1_ids = e1_ids[:self.entity_max_len]
                else:
                    e1_ids += [0] * (self.entity_max_len - len(e1_ids))
                if len(e2_ids) > self.entity_max_len:
                    e2_ids = e2_ids[:self.entity_max_len]
                else:
                    e2_ids += [0] * (self.entity_max_len - len(e2_ids))

                # pos
                e1b, e1e = int(e1_pos.split(':')[0]), int(e1_pos.split(':')[1])
                e2b, e2e = int(e2_pos.split(':')[0]), int(e2_pos.split(':')[1])

                if e1e < e2e:
                    e2es = e2e
                else:
                    e2es = e1e

                lb_dist = []
                for i, _ in enumerate(sent_str):
                    if i in range(e1b, e1e + 1):
                        lb_dist.append(self.limit)
                    elif i < e1b:
                        lb_dist.append(e1b - i + self.limit)
                    elif i > e1e:
                        lb_dist.append(e1b - i + self.limit if e1b - i + self.limit > 0 else 0)

                le_dist = []
                for i, _ in enumerate(sent_str):
                    if i in range(e1b, e1e + 1):
                        le_dist.append(self.limit)
                    elif i < e1b:
                        le_dist.append(e1e - i + self.limit)
                    elif i > e1e:
                        le_dist.append(e1e - i + self.limit if e1e - i + self.limit > 0 else 0)

                rb_dist = []
                for i, _ in enumerate(sent_str):
                    if i in range(e2b, e2e + 1):
                        rb_dist.append(self.limit)
                    elif i < e2b:
                        rb_dist.append(e2b - i + self.limit)
                    elif i > e2e:
                        rb_dist.append(e2b - i + self.limit if e2b - i + self.limit > 0 else 0)

                re_dist = []
                for i, _ in enumerate(sent_str):
                    if i in range(e2b, e2e + 1):
                        re_dist.append(self.limit)
                    elif i < e2b:
                        re_dist.append(e2e - i + self.limit)
                    elif i > e2e:
                        re_dist.append(e2e - i + self.limit if e2e - i + self.limit > 0 else 0)

                # mask
                mask = []
                for i, _ in enumerate(sent_str):
                    if e1b < e2b:
                        if i <= e1e:
                            mask.append(1)
                        elif i <= e2e:
                            mask.append(2)
                        else:
                            mask.append(3)
                    else:
                        if i <= e2e:
                            mask.append(1)
                        elif i <= e1e:
                            mask.append(2)
                        else:
                            mask.append(3)
                ALL = [sent_bert_index, lb_dist, le_dist, rb_dist, re_dist, e1_ids, e2_ids, mask, e2es, e1, e2, sent_str]
                pad_ALL = self.bag_add_bert_pad(ALL)
                if pad_ALL is not None:
                    self.sents.append(pad_ALL)

    def bag_add_bert_pad(self, ALL):
        """添加[PAD], [CLS], [SEP]"""

        def posAddBertPad(L, PAD_NUM):
            new_l = [L[0] + 1] + L + ([L[-1] - 1] if L[-1] - 1 > 0 else [0])
            length = len(new_l)
            new_l = new_l + [0] * PAD_NUM

            for idx in range(length, length + PAD_NUM):
                new_l[idx] = new_l[idx - 1] - 1
            new_l = [_ if _ > 0 else 0 for _ in new_l]
            return new_l

        sentence, lb_dist, le_dist, rb_dist, re_dist, e1, e2, mask, e2es, e1_str, e2_str, sent_str = ALL

        # pad
        if e2es > self.max_len - 2:
            return None
        else:
            # 长截短补
            if len(sentence) > self.max_len:
                sent = sentence[0:self.max_len]
                sent = [self.char2id['[CLS]']] + sent + [self.char2id['[SEP]']]

                lb_dist = lb_dist[0:self.max_len]
                lb_dist = [lb_dist[0] + 1] + lb_dist + ([lb_dist[-1] - 1] if lb_dist[-1] - 1 > 0 else [0])

                le_dist = le_dist[0:self.max_len]
                le_dist = [le_dist[0] + 1] + le_dist + ([le_dist[-1] - 1] if le_dist[-1] - 1 > 0 else [0])

                rb_dist = rb_dist[0:self.max_len]
                rb_dist = [rb_dist[0] + 1] + rb_dist + ([rb_dist[-1] - 1] if rb_dist[-1] - 1 > 0 else [0])

                re_dist = re_dist[0:self.max_len]
                re_dist = [re_dist[0] + 1] + re_dist + ([re_dist[-1] - 1] if re_dist[-1] - 1 > 0 else [0])

                mask = [1] + mask[0:self.max_len] + [0]
            else:
                pad_num = self.max_len - len(sentence)
                sent = [self.char2id['[CLS]']] + sentence + [self.char2id['[SEP]']]
                sent = sent + [self.char2id['[PAD]']] * pad_num

                lb_dist = posAddBertPad(lb_dist, pad_num)
                le_dist = posAddBertPad(le_dist, pad_num)
                rb_dist = posAddBertPad(rb_dist, pad_num)
                re_dist = posAddBertPad(re_dist, pad_num)
                mask = [1] + mask + [0] + [0] * pad_num

        new_ALL = [sent, lb_dist, le_dist, rb_dist, re_dist, e1, e2, mask, e1_str, e2_str, sent_str]

        return new_ALL

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.sents[index]
        word = torch.tensor(sent[0], dtype=torch.long).unsqueeze(0)
        pos1 = torch.tensor(sent[1], dtype=torch.long).unsqueeze(0)
        pos2 = torch.tensor(sent[2], dtype=torch.long).unsqueeze(0)
        pos3 = torch.tensor(sent[3], dtype=torch.long).unsqueeze(0)
        pos4 = torch.tensor(sent[4], dtype=torch.long).unsqueeze(0)
        ent1 = torch.tensor(sent[5], dtype=torch.long).unsqueeze(0)
        ent2 = torch.tensor(sent[6], dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(sent[7], dtype=torch.long).unsqueeze(0)
        e1_str = sent[-3]
        e2_str = sent[-2]
        sent_str = sent[-1]

        return word, pos1, pos2, pos3, pos4, ent1, ent2, mask, e1_str, e2_str, sent_str

    def rel_num(self):
        return len(self.rel2id)


def collate_fn(X):
    X = list(zip(*X))  # 解压
    word, pos1, pos2, pos3, pos4, ent1, ent2, mask, e1_str, e2_str, sent_str = X
    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)

    pos2 = torch.cat(pos2, 0)
    pos3 = torch.cat(pos3, 0)
    pos4 = torch.cat(pos4, 0)
    mask = torch.cat(mask, 0)
    ent1 = torch.cat(ent1, 0)
    ent2 = torch.cat(ent2, 0)

    e1_str = e1_str[0]
    e2_str = e2_str[0]
    sent_str = sent_str[0]

    return word, pos1, pos2, pos3, pos4, ent1, ent2, mask, e1_str, e2_str, sent_str


def label_data_loader(opt, input_path, shuffle, num_workers=0):
    dataset = Label_Dataset(opt, input_path)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader


if __name__ == '__main__':
    opt = DefaultConfig()
    from global_var import transfer_input_path

    data_loader = label_data_loader(opt, transfer_input_path, shuffle=False)
    for _ in data_loader:
        print(_)