import json

file_path = './for_vf_train_data/country of citizenship.json'
output_file = './for_vf_train_data/country of citizenship_new.json'


def remove_duplicate_head_tail(file_path, output_file):
    head_tail_pairs = set()
    unique_lines = []

    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            i += 1
            data = json.loads(line)
            head = data['head']
            tail = data['tail']
            head_tail = (head, tail)
            if head_tail not in head_tail_pairs:
                head_tail_pairs.add(head_tail)
                unique_lines.append(line)

    print(i - len(unique_lines))
    with open(output_file, 'w') as f:
        for line in unique_lines:
            f.write(line)


remove_duplicate_head_tail(file_path, output_file)
