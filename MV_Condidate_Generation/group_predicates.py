import re

from MV_Condidate_Generation.dataset_schema import Dataset_Schema


def GroupPredicates(Predicates_List, Tables_list ,All_Join_Predicates,All_selection_Predicates, index_predicate):
    Join_Predicates = [] # join predicates    table1.key = table2.key
    Selection_Predicates_By_Table = {}# dictionary of table as key and its selection predicates   table.att = value
    dataset_Schema = Dataset_Schema()
    # initialisation of the dico : Selection_Predicates_By_Table
    for table in Tables_list:
        table['name'] = re.sub(r'\d+', '', table['name'])
        if (table['name'] == "miidx"):
            table['name'] = "mi_idx"
        elif (table['name'] == "a"):
            table['name'] = "an"
        index_of_table = dataset_Schema[table['name']]
        Selection_Predicates_By_Table[index_of_table] = []

    for tuple in Predicates_List:
        # Predicates_List is a list of selection predicates represented by tuples of the forme (left side, operator, right side)
        left_table = str(tuple[0]).split('.')[0]
        right_table = str(tuple[2]).split('.')[0]

        if (left_table in (table['name'] for table in Tables_list)) and (right_table  in (table['name'] for table in Tables_list)):
            #This is a join predicat
            if tuple not in All_Join_Predicates:
                All_Join_Predicates.append(tuple)
        else:
            x = [All_selection_Predicates[y] for y in All_selection_Predicates.keys()]
            #print("line29 group_predicats:",All_selection_Predicates)
            if tuple not in x :
                    s = 's' + str(index_predicate)
                    All_selection_Predicates[s] = tuple
                    index_predicate+=1
                    #print("line33 from group_predicates.py", tuple)
                    left_table = re.sub(r'\d+', '', left_table)
                    if (left_table in (table['name'] for table in Tables_list)):
                        index_table = dataset_Schema[left_table]  # get the index of the left table in the tables coding schema
                        Selection_Predicates_By_Table[index_table].append(s)
            else:
                # get its id si
                index_existing_predicate = [key for key, value in All_selection_Predicates.items() if value == tuple]
                left_table = re.sub(r'\d+', '', left_table)
                if (left_table in (table['name'] for table in Tables_list)):
                    index_table = dataset_Schema[left_table]  # get the index of the right table in the tables coding schema
                    Selection_Predicates_By_Table[index_table].append(index_existing_predicate[0])

    return  All_Join_Predicates, Selection_Predicates_By_Table,All_selection_Predicates,index_predicate