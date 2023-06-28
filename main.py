import pprint
import random

import Database
from Benefit_Estimation_Model.Encoder.data_preparation import data_preparation
from Benefit_Estimation_Model.Encoder.model import Encoder
from Benefit_Estimation_Model.Reducer.model import Reducer
from Benefit_Estimation_Model.plan_extraction import getCostPlanJson
from Benefit_Estimation_Model.sql2fea import TreeBuilder, ValueExtractor
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("START AUTOVIEW")
    startime = time.time()
    connexion = Database.connect()
    Path_Workload = '/Users/ilyes/Downloads/RTOS_EXPLICIT_JOIN'
    # Path_Workload = '/Users/ilyes/Downloads/queries/ii-113'
    # Path_Workload = '/Users/ilyes/Downloads/queries/sa-113'
    #Path_Workload = '/Users/ilyes/Downloads/extended_job_queries_solutions'

    # # # PARSING==================================================================
    List_All_queries, List_All_queries_with_their_Tables, List_All_queries_with_their_Tables_and_their_Predicates, \
        All_Join_Predicates, All_selection_Predicates, Workload_Size, \
        List_All_queries_with_their_parsed_format ,All_queries_with_id_query ,Queries_with_id_and_filename= ParseWorkload(Path_Workload)


    # DBMS COST ESTIMATION================================================================
    Rewritten_List_All_queries_with_their_EstimatedCost, Rewritten_Total_Queries_Estimated_Cost = GetQueriesEstimatedCost(
        List_All_queries,
        connexion)


    Dic_Query_With_Query_Join_tree_graph = {}
    Dic_Query_With_Query_Join_tree_graph_Light = {}
    Dic_Query_by_Oreder_In_The_Workload = {}

    startime_generate_query_trees = time.time()
    i = 0

    ListQueries = list(List_All_queries_with_their_Tables.keys())
    for query in List_All_queries_with_their_Tables:
        Dico_query_tables_and_predicates = dict(List_All_queries_with_their_Tables_and_their_Predicates[query])
      #  Dico_query_tables_and_selectAttributes = dict(List_All_queries_with_their_Select_Attributes[query])
        query_join_order = List_All_queries_with_their_Tables[query]

        Query_Join_tree_graph = join_graph.Create_Graph(query,
                                                        Dico_query_tables_and_predicates,
                                                        query_join_order)
        # Path_to_MVPP = '/Users/ilyes/Downloads/gml/' + str(i) + '.gml'
        # nx.write_gml(Query_Join_tree_graph, Path_to_MVPP)

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

   # Path_to_MVPP = '/Users/ilyes/Downloads/gml/MVPP' + str(i) + '.gml'
   # nx.write_gml(MVPP, Path_to_MVPP)

    # MVPP GRAPH VISUALISATION
    # nx.draw(MVPP,with_labels=True)
    # plt.draw()
    # plt.show()

    # COST COMPUTING
    MVPP_With_Selection_0_With_Cost, List_Nodes_With_SQL_Script = computing_cost.ComputeCost(
        MVPP,
        All_Join_Predicates,
        All_selection_Predicates,

        Dataset_Schema,
        Dataset_Schema2,
        connexion)

    #Calculating frequency
    frequency,queries_with_corresponding_views = getFrequency(MVPP, Dic_Query_With_Query_Join_tree_graph)

    #Getting the views each with their cost
    views_with_cost = Get_Views_Info_From_MVPP(MVPP_With_Selection_0_With_Cost,
                                               frequency,
                                               ListQueries)
    print("views_with_cost:",views_with_cost)

    MV_Condidates = {}
    for v in views_with_cost:
            views_with_cost[v].append({"Frequency":frequency[v] } )
    print("views_with_cost:", views_with_cost)


    #Ordering the views on frequency
    views_with_cost = OrderedDict(sorted(views_with_cost.items(), key=lambda x: x[1][3]["Frequency"] , reverse=True))

    for v in views_with_cost.items():
        if (v[1][0] != 0):
            MV_Condidates[v[0]] = v[1]

    #Getting the 10 best views
    MV_Condidates = dict(list(MV_Condidates.items())[:10])

    #Getting the sql script of the views
    for v in MV_Condidates:
        MV_Condidates[v].append(List_Nodes_With_SQL_Script[v])

    print("THE BEST MV Condidates (Sorted from best to worst) :======================================================================")
    for v in MV_Condidates:
        print(v , " : ", MV_Condidates[v])

    nb_of_optimized_queries = 0

    #Calculating the total number of optimized queries
    for frq in frequency.values():
        if frq > 1:
            nb_of_optimized_queries += frq


    #Printing results
    print("TOTAL NUMBER OF OPTIMIZED QUERIES:", nb_of_optimized_queries)
    views_to_be_materialized = []
    for v in MV_Condidates:
        print("VIEWS TO BE MATERIALIZED:",v, MV_Condidates[v][4])
        views_to_be_materialized.append(v)



    # ================BENEFIT ESTIMATION PHASE===============================================================

    # ================ENCODER===============================================================================
   # Encoder(connexion,Path_Workload)



    # ================REDUCER===============================================================================
    Reducer(connexion,MV_Condidates,queries_with_corresponding_views,All_queries_with_id_query,views_to_be_materialized ,Queries_with_id_and_filename)