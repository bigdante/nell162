from utils import *

if __name__ == '__main__':
    # mode = "baseline"
    mode = "alias"
    base_file_path = "./make_COT_traindata_redocred/baseline/query_data" if mode == "baseline" else "./make_COT_traindata_redocred/alias/query_data"
    out_file_path = "./make_COT_traindata_redocred/baseline/query_result" if mode == "baseline" else "./make_COT_traindata_redocred/alias/query_result"
    for relation in relation_list[:1]:
        file_path = os.path.join(base_file_path, relation + ".json")
        out_path = os.path.join(out_file_path, relation + ".json")
        try:
            data = get_uncompleted_data(file_path, out_path)
            if data:
                print(f"{relation} remain {len(data)} to query")
            else:
                print(f"{relation} {relation_list.index(relation)} completed")
                continue
        except FileNotFoundError:
            print(f"File '{out_path}' not found, query begin")
            data = open(file_path).readlines()
        process_all_data(data, relation, mode)
