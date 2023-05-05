import json

# 读取query_data和query_result文件夹下的country of citizenship.json文件
from huggingface_hub.utils import tqdm

with open('query_data/country of citizenship.json', 'r') as f:
    data = f.readlines()

with open('query_result/country of citizenship.json', 'r') as f:
    result = f.readlines()

# 将每一行转换为JSON格式
data = [json.loads(line) for line in data]
result = [json.loads(line) for line in result]

# 遍历data列表，对于每一行，根据uuid在result列表中找到对应的行，提取其tail并添加到data行中
for i in tqdm(range(len(data))):
    uuid = data[i]['uuid']
    for j in range(len(result)):
        if result[j]['input']['uuid'] == uuid:
            result[j]['tail'] = data[i]['tail']
            break

# 将更新后的data列表中的每一行写入到新的文件country of citizenship_test.json中
with open('country of citizenship_test.json', 'w') as f:
    for line in result:
        # 只提取"result": "false_p"和"result": "true_p"的行
        if line['result'] in ['false_p', 'true_p']:
            f.write(json.dumps(line) + '\n')
