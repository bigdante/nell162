import json

# 打开key.txt文件并逐行读取
with open('120-1.txt', 'r') as f:
    lines = f.readlines()

data = {}
# 遍历每行数据，提取key
for id,line in enumerate(lines):
    key = line.strip().split('|')[2]
    data[key] = True

# 保存为json文件
with open('120_key1.json', 'w') as f:
    json.dump(data, f)
