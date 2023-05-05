import json

with open('country of citizenship.json', 'r') as f_in, open('country of citizenship_test.json', 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        head = data['head']
        tail = data['vote_answer']
        result = data['result']
        if result == 'true_p' or result == 'false_p':
            new_data = {'head': head, 'tail': tail, 'result': result}
            json.dump(new_data, f_out)
            f_out.write('\n')
