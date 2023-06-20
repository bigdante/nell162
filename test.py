a = ['a', 'b']
result = {}
for i in a:
    result['relations'] = result.get("relation", []).append(i)
    print(result)
