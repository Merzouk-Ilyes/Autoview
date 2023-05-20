import time
import os

def getCostPlanJson(connexion,path_workload):
    queries= []
    plans= []
    for filename in os.listdir(path_workload):
        if filename.endswith('.sql'):
            with open(os.path.join(path_workload, filename), 'r') as file:
                # Read the queries
                query = file.read()
                queries.append(query)


    cur = connexion.cursor()
    startTime = time.time()
    cur.execute("SET join_collapse_limit = 1 ;")
    #cur.execute("SET geqo_threshold  = 12;")
    for q in queries:
        cur.execute("explain  (COSTS, FORMAT JSON) " + q)
        rows = cur.fetchall()
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time() - startTime
        plans.append(plan_json)

    return plans,queries