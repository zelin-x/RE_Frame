import re

sent_str = '高血压危象治疗一般考虑持续稳定剂量药物的静脉滴入或泵注，常用的药物有硝普纳、硝酸甘油、尼卡地平或拉贝洛尔。'
sent_split = re.split('、|，|；|;|和|或|,|。', sent_str)
print(sent_split)