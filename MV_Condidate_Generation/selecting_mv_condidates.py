import networkx as nx
from collections import OrderedDict

#
# def MVSelection_Heuristic_algorithm(MVPP_With_Selection_With_Cost,
#                                     List_All_queries_Frequencies,
#                                     ListQueries):
#
#     Dico_of_views, Queies_by_views, list_optimized_queries11 = Get_Views_Info_From_MVPP(MVPP_With_Cost,
#                                                                                         List_All_queries_Frequencies,
#                                                                                         ListQueries)
#
#
#     views_with_positif_weight = {}
#     for view in Dico_of_views:
#         if float(Dico_of_views[view][4]) > 0:  # weight positif
#             views_with_positif_weight[view] = float(Dico_of_views[view][4])
#
#     # for view in views_with_positif_weight:
#     # print('=======8=======',view, views_with_positif_weight[view])
#     # print('Dico_of_views ', len(Dico_of_views), 'views_with_positif_weight ', len(views_with_positif_weight))
#
#     # print('views_with_weight :',views_with_weight )
#     views_ordered_by_positif_weight = OrderedDict(
#         sorted(views_with_positif_weight.items(), key=lambda x: x[1], reverse=True))
#     # print('views_ordered_by_positif_weight  :',views_ordered_by_positif_weight )
#
#     # print('views_ordered_by_positif_weight ')
#     # print(views_ordered_by_positif_weight)
#
#     MV = {}  # view with its cost
#
#     # remove views with negatif weight
#
#     lit_view_to_delete_from_views_ordered_by_weight = []
#
#     for view in views_ordered_by_positif_weight:
#         # print('view ', view)
#         # print('lit_view_to_delete_from_views_ordered_by_weight: ',lit_view_to_delete_from_views_ordered_by_weight)
#         if view not in lit_view_to_delete_from_views_ordered_by_weight:  #
#             # print('tttttttttttttttttttttttttttttttt ')
#             view_cost = 0
#             # print(view,':',views_ordered_by_weight[view])
#             list_info_view = list(Dico_of_views[view])
#             # view_queries_access_cost = int(list_info_view[0])
#             # view_maintenace_cost = int(list_info_view[1])
#             list_previous_view = list(list_info_view[2])
#
#             # view_cost = view_queries_access_cost - view_maintenace_cost
#
#             weight = float(list_info_view[4])
#
#             # print('==================================================')
#             # print(view)
#             # print(view_queries_access_cost)
#             # print(view_maintenace_cost)
#             # print(list_previous_view)
#             # print('==================================================')
#             # print('list_previous_view ',list_previous_view)
#             for view_descent in list_previous_view:
#
#                 if view_descent in MV.keys():
#                     list_info_view_temp = list(Dico_of_views[view_descent])
#                     ###########################################################
#                     weight = weight - float(list_info_view_temp[0])
#                     ###########################################################
#             # print('xxxxxxxx  ',view, weight)
#             if weight > 0:
#                 # print('hooooo')
#                 # step 6 in Yang Algorithm
#                 MV[view] = weight
#                 lit_view_to_delete_from_views_ordered_by_weight.append(view)
#                 # print('lit_view_to_delete_from_views_ordered_by_weight ', lit_view_to_delete_from_views_ordered_by_weight)
#             else:
#                 # print('heeeeee')
#                 # step 7 in Yang Alorithm
#                 lit_view_to_delete_from_views_ordered_by_weight.append(view)
#                 # print('lit_view_to_delete_from_views_ordered_by_weight ',lit_view_to_delete_from_views_ordered_by_weight)
#                 # delete the nodes in the same branch as that view
#                 # descendants = list(nx.descendants(MVPP_With_Cost, view))
#                 ancestors = list(nx.ancestors(MVPP_With_Cost, view))
#                 for ancestor in ancestors:
#                     if ancestor in MV.keys():
#                         weight1 = float(Dico_of_views[view][4])
#                         weight2 = float(Dico_of_views[ancestor][4])
#                         if weight2 < weight1:
#                             lit_view_to_delete_from_views_ordered_by_weight.append(ancestor)
#                             # print('lit_view_to_delete_from_views_ordered_by_weight ',lit_view_to_delete_from_views_ordered_by_weight)
#         else:
#             print('9999999999999999')
#     # print('lit_view_to_delete_from_views_ordered_by_weight ', lit_view_to_delete_from_views_ordered_by_weight)
#     # quit()
#     # print('MV',MV)
#
#     # step 9 in Yang Algorithm
#     view_to_delete_V2 = []
#     Final_MV = MV.copy()
#     for mv in MV.keys():
#         # list_info_view = list(Dico_of_views[mv])
#         # list_previous_view = list(list_info_view[2])
#         # print('mv : ',mv,'      list_previous_view ;',list_previous_view )
#
#         views_that_succeed_view = list(MVPP_With_Cost.successors(mv))
#         copy_of_views_that_succeed_view = [ele for ele in views_that_succeed_view if not str(ele).startswith('Q')]
#         # print('mv :',mv)
#         # print('views_that_succeed_view  :',views_that_succeed_view )
#         # print('copy_of_views_that_succeed_view  : ',copy_of_views_that_succeed_view )
#
#         if len(copy_of_views_that_succeed_view) > 0:
#             materilized = True
#             for view in copy_of_views_that_succeed_view:
#                 if view not in MV.keys():
#                     # print('not exist in MV')
#                     materilized = False
#
#             if materilized == True:
#                 # print('removed')
#                 view_to_delete_V2.append(mv)
#         # else:
#         # print('no succserorfor that mv')
#
#     view_to_delete_V2 = set(view_to_delete_V2)
#     # print('view_to_delete : ', view_to_delete_V2)
#     # print('before :', len(MV))
#
#     for mv in view_to_delete_V2:
#         del MV[mv]
#     # end step 9 in Yang algorithm =================
#
#     # print('Final_MV :',MV)
#     # print('after :',len(MV),' + ', len(view_to_delete_V2))
#
#     Queies_by_views_tmp = Queies_by_views.copy()
#     Queies_by_views.clear()
#     # list_optimized_queries_tmp = list_optimized_queries.copy()
#     # list_optimized_queries.clear()
#     list_optimized_queries = []
#
#     total_pages_materialized = 0
#     cost_materialization = 0
#     for mv in MV:
#         total_pages_materialized = total_pages_materialized + Dico_of_views[mv][3]  # nb pages generated
#         cost_materialization = cost_materialization + Dico_of_views[mv][0] + Dico_of_views[mv][1]  # view mainenace cost
#
#         # print('=== : ', Dico_of_views[mv][0] , Dico_of_views[mv][1])
#
#         Queies_by_views[mv] = Queies_by_views_tmp[mv]
#         list_optimized_queries.extend(Queies_by_views_tmp[mv])
#
#     # print('Queies_by_views ', Queies_by_views)
#
#     # print('total_pages_materialized :', total_pages_materialized, ' cost_materialization: ', cost_materialization)
#     # print('best_cost_materialization : ',best_cost_materialization)
#
#
#     views_ordered_by_positif_weight.clear()
#     views_with_positif_weight.clear()
#     # print('best_Dico_of_views ', best_Dico_of_views)
#     # quit()
#     return best_MV, best_MVPP, best_Dico_of_views, best_Queies_by_views, best_list_optimized_queries


def Get_Views_Info_From_MVPP(MVPP_With_Cost, List_All_queries_Frequencies, ListQueries):
    DiconbRowsGenerated = nx.get_node_attributes(MVPP_With_Cost, 'nbRowsGenerated')
    DiconbPageAccessed = nx.get_node_attributes(MVPP_With_Cost, 'nbPageAccessed')
    DiconbPageGenerated = nx.get_node_attributes(MVPP_With_Cost, 'nbPageGenerated')
    Dico_of_views1 = {}
    Queies_by_views1 = {}
    # List_Views : contains all join nodes in the MVPP
    List_Views = [n for n in MVPP_With_Cost.nodes() if str(n).__contains__('_J_')]
    for view in List_Views:
        if view in DiconbPageAccessed and DiconbPageGenerated and DiconbRowsGenerated:
            Dico_of_views1[view] = []
            Queies_by_views1[view] = []
            view_creation_cost = {"Total cost": float(DiconbPageAccessed[view])}
            view_size_in_pages =  {"Size in pages":float(DiconbPageGenerated[view])  }
            view_size_in_rows = {"Size in rows":float(DiconbRowsGenerated[view])}
            # [0] view creation cost [1]: size in pages , [2]: size in rows

            Dico_of_views1[view].append(view_creation_cost)
            Dico_of_views1[view].append(view_size_in_pages)
            Dico_of_views1[view].append(view_size_in_rows)



    return Dico_of_views1
