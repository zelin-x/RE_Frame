关系抽取:(中文电子病历、医学书籍、网络数据爬取)
====

Follow:
    
1. Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks(https://aclanthology.org/D15-1203.pdf)


2. Self-Attention Enhanced Selective Gate with Entity-Aware Embedding for Distantly Supervised Relation Extraction(https://arxiv.org/abs/1911.11899)

Requirement:
======
	Python: 3.8.1
	PyTorch: 1.6.0

Input format:
======
Format1: Like `Input/Input.txt`

`ent1=ent1_type&beg:end ent2=ent2_type&beg:end  NA(fake label)  sentence`

    肝包囊虫病=疾病&15:19	经皮、经肝穿刺置管引流术=检查&0:11	NA	经皮、经肝穿刺置管引流术@3.肝包囊虫病继发感染。

Format2: Like `Input/raw.txt`

`ent1=ent1_type&beg:end ent2=ent2_type&beg:end  ... entn=entn_type&beg:end  sentence`

    经皮肾镜=检查&3:6	婴幼儿=特定人群&12:14	输尿管镜=检查&21:24	内镜=治疗&36:37	肾结石经皮肾镜取石术@对婴幼儿经皮肾镜或经输尿管镜途径碎石，因暂无合适的内镜设备，目前还没能广泛地开展。


Pretrained Embeddings:
====

Download Bert-base-chinese from (https://huggingface.co/bert-base-chinese)


How to run the code?
====
1. Download the pretrained and put them in the `bert_chinese` folder.
2. Add trained models to `checkpoints` folder
3. Add NER result and modify them to the input format like `Input/repo.txt`
4. `python main_seg_one.py` (no other parameters)
5. Get your RE results from `Output/extracted_triples.txt`

