import ast

def get_functions_from_file(file_path):
    with open(file_path, "r") as f:
        file_content = f.read()
    module_node = ast.parse(file_content)
    function_nodes = [node for node in module_node.body if isinstance(node, ast.FunctionDef)]
    function_names = [func.name for func in function_nodes]

    return function_names


file_path = "tool/api.py"
functions = get_functions_from_file(file_path)
function_objs = [getattr(__import__(file_path[:-3].replace("/", ".")), name) for name in functions]



def get_api(api_name: str, *args, **kwargs):
    for name, func in zip(functions, function_objs):
        if name == api_name:
            return func(*args, **kwargs)
    print(f"No such function {api_name}")


# if __name__ == '__main__':
#     for f in functions:
#         print(get_api(f))
