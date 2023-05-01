import pprint

import Database
from Benefit_Estimation_Model.encoding import Encoding
from MV_Condidate_Generation import join_graph, computing_cost
from MV_Condidate_Generation.get_frequency import getFrequency
from MV_Condidate_Generation.merge_plan import MergePlan
from MV_Condidate_Generation.query_estimated_cost import GetQueriesEstimatedCost
from MV_Condidate_Generation.workload import ParseWorkload
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from MV_Condidate_Generation.dataset_schema import Dataset_Schema, Dataset_Schema2
from MV_Condidate_Generation.selecting_mv_condidates import Get_Views_Info_From_MVPP

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("START AUTOVIEW")
    startime = time.time()
    connexion = Database.connect()
    Path_Workload = '/Users/ilyes/Downloads/JOB_PROD'

    # PARSING==================================================================
    List_All_queries, List_All_queries_with_their_Tables, List_All_queries_with_their_Tables_and_their_Predicates, \
        All_Join_Predicates, All_selection_Predicates, List_All_queries_with_their_Select_Attributes, Workload_Size, \
        List_All_queries_with_their_parsed_format = ParseWorkload(Path_Workload)

    # DBMS COST ESTIMATION================================================================
    Rewritten_List_All_queries_with_their_EstimatedCost, Rewritten_Total_Queries_Estimated_Cost = GetQueriesEstimatedCost(
        List_All_queries,
        connexion)

    # for each query generate two tree : one with selection predicat and on called light whiout selection predicates
    # according to the optimal join order extracted from postgresql
    Dic_Query_With_Query_Join_tree_graph = {}
    Dic_Query_With_Query_Join_tree_graph_Light = {}
    Dic_Query_by_Oreder_In_The_Workload = {}

    startime_generate_query_trees = time.time()
    i = 0

    ListQueries = list(List_All_queries_with_their_Tables.keys())
    for query in List_All_queries_with_their_Tables:
        print(query)
        # print(List_All_queries_with_their_Join_Order[query])
        Dico_query_tables_and_predicates = dict(List_All_queries_with_their_Tables_and_their_Predicates[query])
        Dico_query_tables_and_selectAttributes = dict(List_All_queries_with_their_Select_Attributes[query])
        query_join_order = List_All_queries_with_their_Tables[query]

        # print("line52" ,Dico_query_tables_and_selectAttributes)

        Query_Join_tree_graph = join_graph.Create_Graph(query,
                                                        Dico_query_tables_and_predicates,
                                                        query_join_order)
        Path_to_MVPP = '/Users/ilyes/Downloads/gml/' + str(i) + '.gml'
        nx.write_gml(Query_Join_tree_graph, Path_to_MVPP)

        Dic_Query_With_Query_Join_tree_graph[query] = Query_Join_tree_graph
        Dic_Query_by_Oreder_In_The_Workload[query] = i
        i += 1
    endtime_generate_query_trees = time.time() - startime_generate_query_trees

    # GRAPH VISUALISATION
    # nx.draw(Query_Join_tree_graph,with_labels=True)
    # plt.draw()
    # plt.show()

    # THE MERGING PHASE
    Dic_Id_With_MVPP_graph = {}
    # the folowwing loop performs the rotation of query graph for merging
    t1 = OrderedDict(sorted(Dic_Query_by_Oreder_In_The_Workload.items(), key=lambda x: x[1]))
    lst = list(Dic_Query_With_Query_Join_tree_graph.keys())
    i = 0

    Queries_Order_For_Merging = list(t1.keys())
    MVPP = MergePlan(Queries_Order_For_Merging, Dic_Query_With_Query_Join_tree_graph)

    Path_to_MVPP = '/Users/ilyes/Downloads/gml/MVPP' + str(i) + '.gml'
    nx.write_gml(MVPP, Path_to_MVPP)

    # MVPP GRAPH VISUALISATION
    # nx.draw(MVPP,with_labels=True)
    # plt.draw()
    # plt.show()

    # COST COMPUTING
    MVPP_With_Selection_0_With_Cost, List_Nodes_With_SQL_Script = computing_cost.ComputeCost(
        MVPP,
        All_Join_Predicates,
        All_selection_Predicates,
        Dico_query_tables_and_selectAttributes,
        Dataset_Schema,
        Dataset_Schema2,
        connexion)

    frequency = getFrequency(MVPP, Dic_Query_With_Query_Join_tree_graph)

    views_with_cost = Get_Views_Info_From_MVPP(MVPP_With_Selection_0_With_Cost,
                                               frequency,
                                               ListQueries)

    MV_Condidates = {}
    for v in views_with_cost:
        if v in frequency:
            views_with_cost[v].append(frequency[v])

    views_with_cost = OrderedDict(sorted(views_with_cost.items(), key=lambda x: x[1][0]))

    for v in views_with_cost.items():

        if (v[1][0] != 0):
            MV_Condidates[v[0]] = v[1]

    MV_Condidates = dict(list(MV_Condidates.items())[:5])

    for v in MV_Condidates:
        MV_Condidates[v].append(List_Nodes_With_SQL_Script[v])

    print("THE BEST MV Condidates (Sorted from best to worst) :======================================================================")
    for v in MV_Condidates:
        print(v , " : ", MV_Condidates[v])

    #================BENEFIT ESTIMATION PHASE===============================================================

    #================ENCODING===============================================================================
    #
    # Encoding(connexion)