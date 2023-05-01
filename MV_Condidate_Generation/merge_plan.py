import networkx as nx
import queue as q
def MergePlan(Queries_Order_For_Merging, Dic_Query_With_Query_Join_tree_graph):
    first =True
    MVPP = nx.DiGraph()
    print('merge plan:',Queries_Order_For_Merging )
    print('merge plan:',Dic_Query_With_Query_Join_tree_graph )
    for query in Queries_Order_For_Merging:
        gg = Dic_Query_With_Query_Join_tree_graph[query]
        if first:
            MVPP = gg
            first= False
        else:
            MVPP = nx.compose(MVPP, gg)
    return MVPP

# def MergePlan(Queries_Order_For_Merging, Dic_Query_With_Query_Join_tree_graph):
#     S = nx.DiGraph() #MVPP the merged graph
#
#     for query_plan in Dic_Query_With_Query_Join_tree_graph:
#         queue = q.Queue()
#         T = Dic_Query_With_Query_Join_tree_graph[query_plan]
#         S = nx.compose(S,T)
#         for node_g in S:
#             for node_q in T:
#                 if node_q in node_g:
#                     queue.put((node_q,node_g))
#         while queue.qsize() != 0:
#             node_q , node_g = queue.get() #first element in queue
#             #print("line 28" , queue.get())
#             if(nx.is_isomorphic(node_q,node_g)):
#                 queue.put(node_q,node_g)
#                 node_g.add_edges_from(node_q)
#
#     return S