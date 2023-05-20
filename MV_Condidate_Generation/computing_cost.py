import networkx as nx
import re
from MV_Condidate_Generation.subtree_estimated_cost import GetSubTreeEstimatedCost_V1


def ComputeCost(MVPP,
                All_Join_Predicates,
                All_selection_Predicates,
                List_All_queries_with_their_Select_Attributes,
                DW_schema,
                DW_schema2,
                connexion):
    operator_map = {
        "eq": "=",
        "neq": "!=",
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "like": "LIKE",
       "nlike": "NOT LIKE",
        "not_like": "NOT LIKE",
        "in": "IN",
        "between": "BETWEEN",
        "exists" : "EXISTS"
    }

    List_Nodes_With_SQL_Script = {}
    List_nodes = list(MVPP.nodes())

    # print("List_All_queries_with_their_Select_Attributes" ,List_All_queries_with_their_Select_Attributes)
    # get all inner nodes  join nodes and selection nodes
    #List_nodes = [n for n in List_nodes ]
    for node in List_nodes:  # for each inner node


        node_contains = str(node).split('_J_')

        # each node is a candidate node
        if len(node_contains) > 1:  # check if a node is a join node at least two tables
            joined_tables_in_node = []
            joined_tables_in_node_for_sql_query = []

            join_predicates_in_node = []
            all_conjunction_clause = []
            all_projection_attributes = []

            table_conjunction_clause = []
            table_projection_attributes = []
            # we can have id_table concatenate with 'X', we must delete X from the id of the table
            #id_table_without_X = list(id_table.split('-'))[0]

            # decode the id of the table to get the name: get the name of the table
            joined_tables_in_node = [key for key, value in DW_schema().items() if value in node_contains]
            joined_tables_in_node_for_sql_query = [key for key, value in DW_schema2().items() if value in node_contains]

            # table to join
            #joined_tables_in_node.append(Table_Name)
            print("line 58:", joined_tables_in_node,joined_tables_in_node_for_sql_query)

            #joined_tables_in_node_for_sql_query.append(tables_names_for_sql_query)

            # even if the table has non selection predicates, add its join predicates
            # get join  predictaes of the table
            for join_predicate in All_Join_Predicates:
                if str(join_predicate[2]).split(".")[0] in joined_tables_in_node and str(
                    join_predicate[0]).split(".")[0] in joined_tables_in_node:
                    # decode the join predicate from tuple to string: example of join predicate: ('lineorder.lo_custkey', 'eq', 'customer.c_custkey')
                    operator1 = operator_map[join_predicate[1]]
                    join_predicate_to_add = join_predicate[0] + ' ' + operator1 + ' ' + join_predicate[2]
                    join_predicates_in_node.append(join_predicate_to_add)

            join_predicates_in_node = list(set(join_predicates_in_node))

            # get the selection predicates
            # get the successors of the table in the join node
            successors1 = []
            selections = str(node).split('-')


            successors1 = getSelections(selections,successors1)
            # for s in selections:
            #     if "_J_" not in s:
            #         successors1.append(s)
            if len(successors1) > 0:  # table filtred
                #list_predicates = str(successors1[0]).split('-')

                for id_predicate in successors1:  # for each id pedicates get its predicate as tuple and decode it
                    predicate_as_tuple = All_selection_Predicates[id_predicate]
                    operator = operator_map[predicate_as_tuple[1]]


                    if ('.' not in str(predicate_as_tuple[2]) and operator != "IN"):
                        predicate_as_tuple2 =  "'" + str(predicate_as_tuple[2]) + "'"
                    elif (operator == 'IN' and str(predicate_as_tuple[2]).startswith('[')):
                        predicate_as_tuple2 = str(predicate_as_tuple[2]).replace('[', '(').replace(']', ')')
                    elif (operator == 'IN'):
                        predicate_as_tuple2 = "('"+ str(predicate_as_tuple[2]) + "')"

                    else:
                        predicate_as_tuple2 = str(predicate_as_tuple[2])

                    if ('.' not in str(predicate_as_tuple[2])  and predicate_as_tuple[0].split('.')[0] in joined_tables_in_node and operator != "BETWEEN" ):

                        predicate_as_string = re.sub(r'\d+', '',  predicate_as_tuple[0]) + ' ' + operator + ' ' + predicate_as_tuple2
                        table_conjunction_clause.append(predicate_as_string)

                # all_conjunction_clause.append('(' + ' or '.join(table_conjunction_clause) + ')')

                    #attribute = predicate_as_tuple[0]
                    # print('attribute : ',attribute)
                    #table_projection_attributes.append(attribute)

                # print(' length of all_conjunction_clause ',len(all_conjunction_clause))
                #all_projection_attributes.extend(table_projection_attributes)
                all_conjunction_clause_not_empty = []
                # for i in range(0,len(table_conjunction_clause)):
                #     table_conjunction_clause[i] = table_conjunction_clause[i].replace('[', '(').replace(']', ')')
                    # if(all_conjunction_clause[i] != "()" and all_conjunction_clause[i]  not in all_conjunction_clause_not_empty ):
                    #     all_conjunction_clause_not_empty.append(all_conjunction_clause[i])

                for i in range(0,len(table_conjunction_clause)):
                    if("<" in table_conjunction_clause[i]):
                        arr = table_conjunction_clause[i].split("<")
                        arr[0] = arr[0] + "::numeric"
                        table_conjunction_clause[i] = '<'.join(arr)
                    if (" = " in table_conjunction_clause[i]):
                        arr = table_conjunction_clause[i].split("=")
                        arr[1] = str(arr[1])
                        table_conjunction_clause[i] = '='.join(arr)

            merged_array = [joined_tables_in_node_for_sql_query[i] + " AS " + joined_tables_in_node[i] for i in
                            range(len(joined_tables_in_node))]
            merged_array = list(dict.fromkeys(merged_array))

            node_sql_script = ""
            if(len(table_conjunction_clause) > 0 and  len(join_predicates_in_node) > 0):
                node_sql_script = "select * " + \
                                  " \n from " + ' , '.join(merged_array) + \
                                  " \n where \n " + \
                                  (' \n and '.join(table_conjunction_clause)  ) + \
                                  (" AND " if len(table_conjunction_clause) > 0 and len(join_predicates_in_node) > 0 else " " ) + \
                                  ( '\n and \n'.join(list(set(join_predicates_in_node))) )
            elif (len(join_predicates_in_node) > 0) :
                print("line143 computing_cost:",join_predicates_in_node)
                node_sql_script = "select * " + \
                                  " \n from " + ' , '.join(merged_array) + \
                                  " \n where \n " + \
                                  ('\n and \n'.join(list(set(join_predicates_in_node))))

            if(str(node_sql_script) != ""):
                List_Nodes_With_SQL_Script[node] = str(node_sql_script).replace('\n', ' ')


                print("VIEW SQL SCRIPT=>", List_Nodes_With_SQL_Script)
                View_Plan_Total_Cost, plan_nb_rows, Plan_Width = \
                    GetSubTreeEstimatedCost_V1(node_sql_script, connexion)
                print("GetSubTreeEstimatedCost_V1:",GetSubTreeEstimatedCost_V1(node_sql_script, connexion))
                # get the total cost, the number rows, the row width
                nbPageAccessed = round(
                    View_Plan_Total_Cost)  # Plan_Total_Cost :  is the sum of all node's total cost
                nbRowsGenerated = int(plan_nb_rows)
                nbPageGenerated = round((plan_nb_rows * Plan_Width) / 8192)
                # print('view size :', nbPageGenerated)

                # update the node in the MVPP by this values
                attrs = {node: {'nbPageAccessed': nbPageAccessed, 'nbRowsGenerated': nbRowsGenerated,
                                'nbPageGenerated': nbPageGenerated}}
                #print('attrs  for predicatees node : ', attrs)
                nx.set_node_attributes(MVPP, attrs)


        else:
            #print('is a selection node ', node)
            pass

    # quit()
    return MVPP, List_Nodes_With_SQL_Script


# def getSelections(selections,successors1):
#         for s in selections:
#             if "_J_" not in s and s.startswith("s"):
#                 successors1.append(s)
#             else:
#                 s = s.split("_J_")
#                 getSelections(s,successors1)
#         return successors1


def getSelections(selections, successors1):
    for s in selections:
        if "_J_"  in s:
            s = s.split("_J_")
            getSelections(s,successors1)
        elif s.startswith("s"):
            successors1.append(s)

    return successors1