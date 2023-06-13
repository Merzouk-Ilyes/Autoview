from networkx import has_path
import re
def getFrequency(MVPP,Query_Graph):
    list_nodes = MVPP.nodes()
    MVPP_nodes = []
    frequency = {}
    query_nodes = []
    # get all inner nodes  join nodes and selection nodes
    #list_nodes = [n for n in list_nodes ]


    for node in list_nodes:
        if "_J_" in node:
            # We need the views have only one JOIN
            count = node.count("J")
            if(count == 1):
                MVPP_nodes.append(node)
    frequency = {index: 0 for index in MVPP_nodes}
    # for graph in list(Query_Graph.values()):
    #     query_nodes.append(max(graph, key=len))
    potential_views = []
    queries_with_corresponding_views = {}
    for node in MVPP_nodes:
        for q in list(Query_Graph.values()):
           # print("line19:",node,node in max(q.nodes(), key=len) , max(q.nodes(), key=len))
            # for q_node in q.nodes():
            #     print("line23:",node ,has_path(MVPP,node,list(q.nodes())[-1]) , list(q.nodes())[-1])
            #     print("q.nodes=>>",list(q.nodes())[-1])
            #     if( has_path(MVPP,node,list(q.nodes())[-1])):
          # print(list(q.nodes())[1])

           if (node ==  list(q.nodes())[1]):
                queries_with_corresponding_views[list(q.nodes())[-1]] = node
                frequency[node] +=1


    print("QUERIES WITH CORRESPONDING VIEWS:",queries_with_corresponding_views)


    return frequency

