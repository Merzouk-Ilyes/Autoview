
from Benefit_Estimation_Model.Reducer.materialization_of_views import Materialize_view

def Reducer(connextion,mv_candidates):

    cursor = connextion.cursor()
    for v in mv_candidates:

        Materialize_view(mv_candidates[v][4],v,cursor)

