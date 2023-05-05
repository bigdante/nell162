from api import *
from prompt import *

if __name__ == '__main__':
    relation = ""
    recall = 0
    while True:
        relation = input("\033[1;32mPlease input a relation or 'exit' to exit the system:\033[0m")
        if relation not in relation_list:
            print(f"{relation} not in relation list, asshole")
            continue
        if relation == "exit":
            print("Exiting system...")
            break
        while True:
            recall_goal = float(input("\033[1;32m Your recall goal (0-1.0): \033[0m"))
            if recall_goal >= 1 or recall_goal <= 0:
                print("stupid recall goal asshole, re_input")
            else:
                break
        while True:
            get_gpt_result(relation)
            alias_performance, recall = check_every_template(relation, old=False)
            if not alias_performance:
                print("\033[1;32malias template are new generated, without output. Show the old ones:\033[0m")
                alias_performance, recall = check_every_template(relation, old=True)
            if recall:
                if recall >= recall_goal:
                    print("\033[1;32m goal achieve \033[0m")
                    break
                else:
                    print("\033[1;32mgoal not achieved, reconstruct alias .......\033[0m")
                    reconstruct_alias(relation, alias_performance)
            else:
                get_gpt_result(relation)
