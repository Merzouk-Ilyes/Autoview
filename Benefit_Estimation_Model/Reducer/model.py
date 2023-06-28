
from Benefit_Estimation_Model.Reducer.materialization_of_views import Materialize_view
from Benefit_Estimation_Model.Reducer.rewriting_queries import Rewrite_query
import time
from MV_Condidate_Generation.query_estimated_cost import json_extract
import json
from yoctopuce.yocto_api import *
from yoctopuce.yocto_power import *

def Reducer(connection,mv_candidates,queries_with_corresponding_views,All_queries_with_id_query,views_to_be_materialized,Queries_with_id_and_filename):
    # Setup Yoctopuce API
    errmsg = YRefParam()

    # Use VirtualHub or direct USB connection
    if YAPI.RegisterHub("http://127.0.0.1:4444/", errmsg) != YAPI.SUCCESS:
        sys.exit("Failed to register hub (%s)" % errmsg.value)

    # Find the Yoctopuce power sensor
    power_sensor = YPower.FirstPower()

    if power_sensor is None:
        sys.exit("No Yoctopuce power sensor found.")

    cursor = connection.cursor()

    #Materializing the candidate views
    for v in mv_candidates:
        Materialize_view(mv_candidates[v][4],v,cursor)
    print("VIEWS TO BE MATERIALIZED:",views_to_be_materialized)

    rewritten_queries =  Rewrite_query(queries_with_corresponding_views,All_queries_with_id_query,mv_candidates,views_to_be_materialized, Queries_with_id_and_filename)

    #print(rewritten_queries)
    plans= []
    benefits = {}
    cur = connection.cursor()
    startTime = time.time()
    cur.execute("SET join_collapse_limit = 1 ;")
    # cur.execute("SET geqo_threshold  = 12;")
    for key,value in rewritten_queries.items():
        #        cur.execute("explain  (ANALYSE ,COSTS, FORMAT JSON) " + q)
        print("==================================")
        benefits[Queries_with_id_and_filename[key]] = []
        print(Queries_with_id_and_filename[key])

      #  GETTING THE QUERY COST
        power_sensor.startDataLogger()
        print(value[1])
        cur.execute("explain  (  COSTS, FORMAT JSON) " + value[1])
        rows = cur.fetchall()
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time() - startTime
        print("THE PLAN:",plan_json['Plan']['Total Cost'])
        query_cost = plan_json['Plan']['Total Cost']

        power_sensor.stopDataLogger()
        energy_consumption_query = power_sensor.get_currentValue()
        print("Energy Consumption:", energy_consumption_query, "W")


        #GETTING THE QUERY/VIEW COST
        power_sensor.startDataLogger()

        cur.execute("explain  (COSTS, FORMAT JSON) " + value[0])
        rows = cur.fetchall()
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time() - startTime
        print("THE PLAN:", plan_json['Plan']['Total Cost'])
        query_view_cost = plan_json['Plan']['Total Cost']

        power_sensor.stopDataLogger()
        energy_consumption_query_view = power_sensor.get_currentValue()
        print("Energy Consumption of view:", energy_consumption_query_view, "W")

        benefits[Queries_with_id_and_filename[key]].append({"Query Cost" : query_cost , "Query/view (" + queries_with_corresponding_views[key] + ") Cost" : query_view_cost , "Benefit" :query_cost - query_view_cost  })
        benefits[Queries_with_id_and_filename[key]].append({"Query Energy Consumption": energy_consumption_query , "Query/view (" + queries_with_corresponding_views[key] + ") Energy Consumption" : energy_consumption_query_view , "Benefit" :energy_consumption_query - energy_consumption_query_view })
        print(benefits)

        plans.append(plan_json)

    with open("B_PG.json", "w") as file:
        # Write the dictionary to the file in JSON format
        json.dump(benefits, file)

   # print(plans)