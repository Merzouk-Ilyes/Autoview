import numpy as np

import Database
from MV_Condidate_Generation.query_estimated_cost import json_extract

def GetSubTreeEstimatedCost_V1(subQuerySqlscript, connexion):
    # print("GetSubTreeEstimatedCost_V1:" , subQuerySqlscript )
    query_cost, JsonPlan_Format, Plan_Rows,Plan_Width = Database.optimizer_cost(connexion, subQuerySqlscript, force_order=True)
    # print("GetSubTreeEstimatedCost_V1:",JsonPlan_Format)
    #Database.disconnect(connexion)
    #print('JsonPlan_Format GetSubTreeEstimatedCost: ',JsonPlan_Format)
    #json_extract : function that extract from the json file all attribute equal to  "Total Cost"

    List_Plan_Node_Total_Cost = json_extract(JsonPlan_Format, "Total Cost")
    List_Plan_Node_Type = json_extract(JsonPlan_Format, "Node Type")


    # if len(List_Plan_Node_Total_Cost) == len(List_Plan_Node_Type):
        # get the total cost only of the scan, hash, and hash join nodes
        # x = [List_Plan_Node_Total_Cost[i] for i in range(0,len(List_Plan_Node_Total_Cost)) if List_Plan_Node_Type[i] not in ('Aggregate,Sort')]
    x = [List_Plan_Node_Total_Cost[i] for i in range(0, len(List_Plan_Node_Total_Cost)) ]
    # else:
    #     print('ERROR')
    #     quit()

    Plan_Total_Cost = np.array(x).sum()

    return Plan_Total_Cost, Plan_Rows, Plan_Width