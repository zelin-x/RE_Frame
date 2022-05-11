from itertools import permutations
import re


def transfer_input_form(relations_type_list, path1, path2):
    """
    :param relations_type_list: KG中的关系类型
    :param path1: 初始输入文件
    :param path2: 模型输入文件
    """

    def add_stop(sent):
        """添加句号"""
        if sent[-1] == '。' or sent[-1] == '.':
            return sent
        else:
            return sent + '。'

    w = open(path2, 'w', encoding='UTF-8')
    with open(path1, 'r', encoding='UTF-8')as f:
        for line in f:
            lin = line.strip()
            lin = lin.split('\t')
            if len(lin) == 2:
                continue
            combines = list(permutations(range(len(lin) - 1), 2))
            for h, t in combines:
                head, tail = lin[h], lin[t]

                if '&' in head.split('=')[0] or '&' in head.split('=')[0]:
                    continue
                else:
                    e1_pos = head.split('&')[-1].split(':')
                    e1_b, e1_e = int(e1_pos[0]), int(e1_pos[1])
                    e2_pos = tail.split('&')[-1].split(':')
                    e2_b, e2_e = int(e2_pos[0]), int(e2_pos[1])
                    if min(abs(e1_b-e2_e), abs(e1_e-e2_b)) > 50:
                        continue
                    if head.split('=')[0] != tail.split('=')[0]:
                        head_type = head.split('&')[0].split('=')[-1]
                        tail_type = tail.split('&')[0].split('=')[-1]
                        if head_type + tail_type in relations_type_list:
                            w.write(head + '\t' + tail + '\t' + 'NA' + '\t' + add_stop(lin[-1]) + '\n')


def revise_disease_tail_type(s, input_type):
    """修正疾病尾实体类型"""
    if s == 'NA':
        return 'NA'
    tail_type = s.split('\\')[-1]
    if input_type == tail_type:
        if s == '疾病\\多发群体\\特定人群':
            return '疾病\\有多发群体\\特定人群'
        else:
            return s
    else:
        # 如果尾实体类型预测为相似实体类型的
        if s == '疾病\\可被治疗\\操作' and input_type == '治疗':
            return '疾病\\可被治疗\\治疗'
        elif s == '疾病\\可被治疗\\治疗' and input_type == '操作':
            return '疾病\\可被治疗\\操作'
        elif s == '疾病\\可被治疗\\药品' and input_type == '药物':
            return '疾病\\可被治疗\\药物'
        elif s == '疾病\\有并发症\\疾病' and input_type == '症状':
            return '疾病\\导致\\症状'
        else:
            return 'NA'


def revise_symptom_tail_type(s, input_type):
    """修正症状尾实体类型"""
    if s == 'NA':
        return 'NA'
    tail_type = s.split('\\')[-1]
    if input_type == tail_type:
        return s
    else:
        if s == '症状\\是临床表现\\疾病' and input_type == '症状':
            return '症状\\伴随\\症状'
        else:
            return 'NA'


def revise_test_tail_type(s, input_type):
    if s == 'NA':
        return 'NA'
    tail_type = s.split('\\')[-1]
    if input_type == tail_type:
        if s == '检查\\是诊断依据\\疾病':
            return '检查\\是诊断所需检查\\疾病'
        else:
            return s
    else:
        if s == '检查\\证实\\症状' and input_type == '疾病':
            return '检查\\是诊断所需检查\\疾病'
        elif s == '检查\\是诊断依据\\疾病' and input_type == '症状':
            return '检查\\证实\\症状'
        else:
            return 'NA'


def check_supervised_pre(s, head_type, tail_type):
    """检查有监督预测是否离谱"""
    pre_head_type, pre_tail_type = s.split('\\')[0], s.split('\\')[-1]
    if s in ['疾病\\有病因\\社会学', '社会学\\是风险因素\\疾病', '疾病\\有严重程度\\严重等级划分', '社会学\\别名\\社会学'] or (
            head_type == pre_head_type and tail_type == pre_tail_type):
        return False  # 这个抽的实在是太离谱直接不要了吧
    if head_type != pre_head_type:
        if head_type in ['药物', '药品', '治疗'] and pre_head_type in ['药物', '药品', '治疗']:
            pass
        elif head_type in ['治疗', '操作'] and pre_head_type in ['治疗', '操作']:
            pass
        else:
            return False
    elif tail_type != pre_tail_type:
        if tail_type in ['药物', '药品', '治疗'] and pre_tail_type in ['药物', '药品', '治疗']:
            pass
        elif tail_type in ['治疗', '操作'] and pre_tail_type in ['治疗', '操作']:
            pass
        else:
            return False
    return True


def write_file(path, l):
    with open(path, 'w', encoding='UTF-8')as f:
        for _ in l:
            f.write(_ + '\n')


def bieming_rule(s, e1_pos, e2_pos):
    """别名得限制条件"""
    feature1 = ['又称', '又叫', '别名', '别称', '学名', '英文名']
    e1_b, e1_e = int(e1_pos.split(':')[0]), int(e1_pos.split(':')[1])
    e2_b, e2_e = int(e2_pos.split(':')[0]), int(e2_pos.split(':')[1])

    if e1_b > e2_b:
        if s[e2_e + 1] in [',', '，'] and s[e2_b - 1] in ['(', '（'] and s[e1_e + 1] in [')', '）']:
            # (A, B) A and B
            return True
        if s[e2_e + 1] in ['(', '（'] and s[e1_b - 1] in ['，', ','] and s[e1_e + 1] in [')', '）']:
            # A(B, C) A and C
            return True
        if (s[e1_b - 1] == '(' or s[e1_b - 1] == '（') and e2_e + 2 == e2_b:
            return True
        middle_str = s[e2_e + 1:e1_b]
    else:
        if s[e1_e + 1] in [',', '，'] and s[e1_b - 1] in ['(', '（'] and s[e2_e + 1] in [')', '）']:
            # (A, B) A and B
            return True
        if s[e1_e + 1] in ['(', '（'] and s[e2_b - 1] in ['，', ','] and s[e2_e + 1] in [')', '）']:
            # A(B, C) A and C
            return True
        if (s[e2_b - 1] == '(' or s[e2_b - 1] == '（') and e1_e + 2 == e2_b:
            # A(B, C) A and B
            return True
        middle_str = s[e1_e + 1:e2_b]
    for _ in feature1:
        if _ in middle_str:
            return True

    return False


def subClassof_rule(s, e1_str, e2_str, rel):
    """subClassof的限制条件"""
    e1, e1_type = e1_str.split('=')
    e2, e2_type = e2_str.split('=')
    e1_type, e1_pos = e1_type.split('&')
    e2_type, e2_pos = e2_type.split('&')
    if e1 in e2:
        return 'NA'
    elif bieming_rule(s, e1_pos, e2_pos):
        return e1_type + '\\别名\\' + e2_type
    return rel


def split_rule(s, e1_str, e2_str, rel):
    e1, e1_type = e1_str.split('=')
    e2, e2_type = e2_str.split('=')
    split_s = re.split('、|，|,|和|或|及|跟|与|以及', s)
    if e1 in split_s and e2 in split_s:
        return 'NA'
    return rel

