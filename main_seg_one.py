"""
    Pipeline relation extraction models
"""

import torch
import torch.nn.functional as F
import os

from tqdm import tqdm

from Net import SeG_ONE
from config import DefaultConfig
from label_Data_loader import label_data_loader

from utils import *
from global_var import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    print("\n===转换输入格式===")
    transfer_input_form(relation_lists, raw_input_path, transfer_input_path)
    print('===输入格式转换完成，目标写在\"Input/input.txt\"===')

    opt = DefaultConfig()

    disease_opt = DefaultConfig()
    disease_opt.root_path = head_disease_relation_dir
    disease_opt.rel_num = 10

    symptom_opt = DefaultConfig()
    symptom_opt.root_path = head_symptom_relation_dir
    symptom_opt.rel_num = 6

    operation_opt = DefaultConfig()
    operation_opt.root_path = head_operation_relation_dir
    operation_opt.rel_num = 6

    test_opt = DefaultConfig()
    test_opt.root_path = head_test_relation_dir
    test_opt.rel_num = 8

    supervised_opt = DefaultConfig()
    supervised_opt.root_path = head_supervised_relation_dir
    supervised_opt.rel_num = 23

    disease_model = SeG_ONE(disease_opt)
    symptom_model = SeG_ONE(symptom_opt)
    operation_model = SeG_ONE(operation_opt)
    test_model = SeG_ONE(test_opt)
    supervised_model = SeG_ONE(supervised_opt)

    if torch.cuda.is_available():
        disease_model = disease_model.cuda()
        symptom_model = symptom_model.cuda()
        operation_model = operation_model.cuda()
        test_model = test_model.cuda()
        supervised_model = supervised_model.cuda()

    print("\n===加载模型参数===")
    disease_model.load_state_dict(torch.load(head_disease_model_path))
    disease_model.eval()
    symptom_model.load_state_dict(torch.load(head_symptom_model_path))
    symptom_model.eval()
    operation_model.load_state_dict(torch.load(head_operation_model_path))
    operation_model.eval()
    test_model.load_state_dict(torch.load(head_test_model_path))
    test_model.eval()
    supervised_model.load_state_dict(torch.load(head_supervised_model_path))
    supervised_model.eval()
    print("===加载模型参数完成===")

    print("\n===构建Data_loader===")
    test_single_loader = label_data_loader(opt, transfer_input_path, shuffle=False)
    print("===数据加载完成===")

    print("\n===抽取三元组===")
    extract_triples = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_single_loader)):
            if torch.cuda.is_available():
                input_data = [x.cuda() for x in data[:-3]]
            try:
                e1_str, e2_str, sent_str = data[-3:]
                e1, e1_type = e1_str.split('=')
                e2, e2_type = e2_str.split('=')
                e1_type, e1_pos = e1_type.split('&')
                e2_type, e2_pos = e2_type.split('&')
            except:
                continue
            sent_split = re.split('、|，|,|和|或|及|跟|与|以及', sent_str)
            if e1 in sent_split and e2 in sent_split:
                continue
            word, pos1, pos2, pos3, pos4, ent1, ent2, mask = input_data
            flag = True
            if e1_type == '疾病':
                out = disease_model(word, pos1, pos2, pos3, pos4, ent1, ent2, mask)
                out = F.softmax(out, 1)
                max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
                pre_class = max_ins_label.tolist()[0]
                relation_str = disease_rel_dic[pre_class]
                DS_pre_rel = revise_disease_tail_type(relation_str, e2_type)
            elif e1_type == '症状':
                out = symptom_model(word, pos1, pos2, pos3, pos4, ent1, ent2, mask)
                out = F.softmax(out, 1)
                max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
                pre_class = max_ins_label.tolist()[0]
                relation_str = symptom_rel_dic[pre_class]
                DS_pre_rel = revise_symptom_tail_type(relation_str, e2_type)
            elif e1_type == '操作':
                out = operation_model(word, pos1, pos2, pos3, pos4, ent1, ent2, mask)
                out = F.softmax(out, 1)
                max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
                pre_class = max_ins_label.tolist()[0]
                relation_str = operation_rel_dic[pre_class]
                DS_pre_rel = relation_str
            elif e1_type == '检查':
                out = test_model(word, pos1, pos2, pos3, pos4, ent1, ent2, mask)
                out = F.softmax(out, 1)
                max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
                pre_class = max_ins_label.tolist()[0]
                relation_str = test_rel_dic[pre_class]
                DS_pre_rel = revise_test_tail_type(relation_str, e2_type)
            else:
                flag = False
            out = supervised_model(word, pos1, pos2, pos3, pos4, ent1, ent2, mask)
            out = F.softmax(out, 1)
            super_max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
            pre_class = max_ins_label.tolist()[0]
            supervised_rel = supervised_rel_dic[pre_class]

            # 对关系进行判定输出
            if flag:
                # 如果头实体不在DS模型中
                if supervised_rel == DS_pre_rel:
                    # 判定一致
                    output_rel = supervised_rel
                else:
                    if supervised_rel == 'NA':
                        if max_ins_prob > 0.8 and DS_pre_rel in ['疾病\\有就诊科室\\科室', '疾病\\可被治疗\\操作',
                                                                 '症状\\可使用检查\\检查', '症状\\有就诊科室\\科室',
                                                                 '症状\\有发生部位\\人体'] + list(operation_rel_dic.values()) + \
                                list(test_rel_dic.values()):
                            # 远程监督预测非NA且是DS模型独有的关系类型
                            output_rel = DS_pre_rel + ' &&DS'
                        else:
                            output_rel = 'NA'
                    elif DS_pre_rel == 'NA':
                        if check_supervised_pre(supervised_rel, e1_type, e2_type) and super_max_ins_prob > 0.8:
                            output_rel = supervised_rel + ' &&SUP'
                        else:
                            output_rel = 'NA'
                    else:
                        # 都不是NA,但预测不一致
                        if check_supervised_pre(supervised_rel, e1_type, e2_type) and super_max_ins_prob > 0.8:
                            # 有监督预测合理且置信度>0.8
                            output_rel = supervised_rel + ' &&SUP'
                        else:
                            output_rel = DS_pre_rel + ' &&DS'
            else:
                if supervised_rel != 'NA' and check_supervised_pre(supervised_rel, e1_type,
                                                                   e2_type) and super_max_ins_prob > 0.8:
                    output_rel = supervised_rel + ' &&SUP'
                else:
                    output_rel = 'NA'

            if '别名' in output_rel:
                if not bieming_rule(sent_str, e1_pos, e2_pos):
                    output_rel = 'NA'
            elif 'subClassof' in output_rel:
                output_rel = subClassof_rule(sent_str, e1_str, e2_str, output_rel)

            # output_rel = split_rule(sent_str, e1_str, e2_str, output_rel)

            if output_rel != 'NA':
                output_triple = e1_str + '\t' + output_rel + '\t' + e2_str + '\t' + sent_str
                extract_triples.append(output_triple)

    # extract_triples = list(set(extract_triples))
    print("===抽取三元组完成===")
    print("\n===抽取三元组写入\"Output/extracted_triples.txt\"===")
    write_file(output_path, extract_triples)
    print("===抽取完成===")
