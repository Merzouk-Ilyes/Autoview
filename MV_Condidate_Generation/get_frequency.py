from networkx import has_path
def getFrequency(MVPP,Query_Graph):
    list_nodes = MVPP.nodes()
    MVPP_nodes = []
    frequency = {}
    query_nodes = []
    # get all inner nodes  join nodes and selection nodes
    #list_nodes = [n for n in list_nodes ]

    for node in list_nodes:
        if "_J_" in node:
            MVPP_nodes.append(node)
    frequency = {index: 0 for index in MVPP_nodes}


    # for graph in list(Query_Graph.values()):
    #     query_nodes.append(max(graph, key=len))

    for node in MVPP_nodes:
        for q in list(Query_Graph.values()):
            print("line19:",node,q.nodes())
            for q_node in q.nodes():
                if( has_path(MVPP,node,q_node)):
                    frequency[node] +=1
    print('line 21 :' , frequency)

    return frequency

