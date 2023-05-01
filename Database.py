import psycopg2
from config import config


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def disconnect(conn):
    if conn is not None:
        conn.close()
        print('Database connection closed.')


def optimizer_cost(conn, query, force_order):
    join_collapse_limit = "SET join_collapse_limit = "
    join_collapse_limit += "1" if force_order else "8"
    # query = "set schema 'AED'; SET max_parallel_workers_per_gather = 0;" + join_collapse_limit + ";EXPLAIN (FORMAT JSON) " + query + ";"
    query = "set schema 'public';" + join_collapse_limit + ";EXPLAIN (FORMAT JSON) " + query + ";"
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchone()

    cursor.close()
    return rows[0][0]["Plan"]["Total Cost"], rows[0][0], rows[0][0]["Plan"]["Plan Rows"], rows[0][0]["Plan"][
        "Plan Width"]
