raw_input_path = r'Input/raw.txt'
transfer_input_path = r'Input/input.txt'

output_path = r'Output/extracted_triples.txt'

head_disease_model_path = r'checkpoints/model_head_disease.pth'
head_disease_relation_dir = r'my_Data/disease_head'

head_symptom_model_path = r'checkpoints/model_head_symptom.pth'
head_symptom_relation_dir = r'my_Data/symptom_head'

head_operation_model_path = r'checkpoints/model_head_operation.pth'
head_operation_relation_dir = r'my_Data/operation_head'

head_test_model_path = r'checkpoints/model_head_test.pth'
head_test_relation_dir = r'my_Data/test_head'

head_supervised_model_path = r'checkpoints/model_huawei4_supervised.pth'
head_supervised_relation_dir = r'my_Data/supervised2'


def get_rel_dic(path):
    dic = {}
    with open(path + '/relation.txt', 'r', encoding='UTF-8')as f:
        for line in f:
            lin = line.strip()
            if lin:
                lin = lin.split()
                dic[int(lin[1])] = lin[0]
    return dic


disease_rel_dic = get_rel_dic(head_disease_relation_dir)
symptom_rel_dic = get_rel_dic(head_symptom_relation_dir)
operation_rel_dic = get_rel_dic(head_operation_relation_dir)
test_rel_dic = get_rel_dic(head_test_relation_dir)
supervised_rel_dic = get_rel_dic(head_supervised_relation_dir)

relation_type_path = r'my_Data/【类型列表】合并版本-关系类型列表.txt'
relation_lists = []
with open(relation_type_path, 'r', encoding='UTF-8')as f:
    for line in f:
        lin = line.strip()
        relation = lin.split('\t')[1]
        h_t, r, t_t = relation.split('\\')
        relation_lists.append(h_t + t_t)

relation_lists = list(set(relation_lists))
