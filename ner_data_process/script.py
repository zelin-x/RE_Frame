sents = []
labels = []
with open('fusion.txt', 'r', encoding='UTF-8')as f:
    sent = []
    label = []
    for line in f:
        lin = line.strip()
        if lin:
            lin = lin.split('\t')
            if lin[0] not in ['。', '!', ';', '；']:
                sent.append(lin[0])
                label.append(lin[2])
            else:
                sents.append(sent)
                labels.append(label)
                sent = []
                label = []

write_sents = []
for i in range(len(sents)):
    sent = sents[i]
    label = labels[i]
    entites = []
    in_entity = False
    for i, _ in enumerate(label):
        if 'B' in _:
            beg = i
            in_entity = True
        elif 'I' in _:
            if not in_entity:
                in_entity = False
                continue
            else:
                continue
        else:
            if in_entity:
                end = i
                entites.append((beg, end - 1))
                in_entity = False
    ss = ''
    for b, e in entites:
        ss += ''.join(sent[b:e + 1]) + '=' + label[b].split('-')[-1] + '&' + str(b) + ':' + str(e) + '\t'

    ss += ''.join(sent) + '\n'
    write_sents.append(ss)

ff = open('./out.txt', 'w', encoding='UTF-8')
for _ in write_sents:
    ff.write(_)
