import networkx as nx


def Create_Graph(query, Dico_query_tables_and_predicates, query_join_order):
    Query_Join_tree_graph = nx.DiGraph()
    Query_Join_tree_graph.clear()
    EdgeToAdd = []
    print("QUERY:",query)
    print("Dico_query_tables_and_predicates:",Dico_query_tables_and_predicates)
    print("query_join_order:",query_join_order)

    first_table = query_join_order[0]
    # clausepredicates = Dico_query_tables_and_predicates[query_join_order[1]]
    # firstJoin = first_table + '_J_' + query_join_order[1] + '-' + '-'.join(clausepredicates)

    clausepredicates = Dico_query_tables_and_predicates[first_table]
    #if (len(clausepredicates) > 0):
    firstJoin = first_table + '-' + '-'.join(clausepredicates)

    if len(clausepredicates) == 1:

        andPredicates = '-'.join(clausepredicates)
        edge = (first_table, andPredicates)
        EdgeToAdd.append(edge)

        edge = (andPredicates, firstJoin)
        EdgeToAdd.append(edge)
    elif len(clausepredicates) > 1:  # test if there is a clause of predicat

        andPredicates = '-'.join(clausepredicates)

        for predicate in range(0, len(clausepredicates)):
            edge = (first_table , clausepredicates[predicate])
            EdgeToAdd.append(edge)
            edge = (clausepredicates[predicate], andPredicates)
            EdgeToAdd.append(edge)


        edge = (first_table ,andPredicates)
        EdgeToAdd.append(edge)

        edge = (andPredicates , firstJoin)
        EdgeToAdd.append(edge)


    # for remaining tables in the join order
    for table in query_join_order[1:]:
        clausepredicates = Dico_query_tables_and_predicates[table]
        if len(clausepredicates) == 1:  # test if there is a clause of predicat
            PreviousJoin = firstJoin
            # firstJoin = firstJoin + '_J_' + table + '-' + '-'.join(clausepredicates)
            firstJoin = PreviousJoin + '_J_' + table + '-' + '-'.join(clausepredicates)


            andPredicates = '-'.join(clausepredicates)

            edge = (table, andPredicates)
            EdgeToAdd.append(edge)

            edge = (andPredicates, firstJoin)
            EdgeToAdd.append(edge)

            edge = (PreviousJoin, firstJoin)
            EdgeToAdd.append(edge)

        if len(clausepredicates) > 1:  # .startswith('cl'):
            PreviousJoin = firstJoin
            # firstJoin = firstJoin + '_J_' + table + '-' + '-'.join(clausepredicates)
            firstJoin = PreviousJoin + '_J_' + table + '-' + '-'.join(clausepredicates)
            andPredicates = '-'.join(clausepredicates)

            for predicate in range(0, len(clausepredicates)):
                edge = (table, clausepredicates[predicate])
                EdgeToAdd.append(edge)
                edge = (clausepredicates[predicate], andPredicates)
                EdgeToAdd.append(edge)

            edge = (andPredicates, firstJoin)
            EdgeToAdd.append(edge)

            edge = (PreviousJoin, firstJoin)
            EdgeToAdd.append(edge)

        if len(clausepredicates) == 0:  # .startswith('cl'):
            PreviousJoin = firstJoin
            firstJoin = PreviousJoin + '_J_' + table
            edge = (PreviousJoin, firstJoin)
            EdgeToAdd.append(edge)
            edge = (table, firstJoin)
            EdgeToAdd.append(edge)

    edge = (firstJoin, query)
    EdgeToAdd.append(edge)
    Query_Join_tree_graph.add_edges_from(EdgeToAdd)


    return Query_Join_tree_graph
