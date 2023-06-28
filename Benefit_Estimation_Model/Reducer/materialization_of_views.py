from psycopg2 import ProgrammingError
from moz_sql_parser import parse
import psycopg2
from psycopg2 import Error


def Materialize_view(view_script, view_name, cursor):
    schema = {
        "title": ["id", "kind_id", "production_year", "imdb_id", "episode_of_id", "season_nr", "episode_nr", "title",
                  "imdb_index", "phonetic_code", "series_years", "md5sum"], "role_type": ["id", "role"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"], "link_type": ["id", "link"],
        "name": ["id", "imdb_id", "name", "imdb_index", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                 "md5sum"], "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "kind_type": ["id", "kind"], "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
        "company_type": ["id", "kind"], "comp_cast_type": ["id", "kind"],
        "person_info": ["id", "person_id", "info_type_id", "note", "info"],
        "aka_title": ["episode_of_id", "season_nr", "episode_nr", "id", "kind_id", "production_year", "movie_id",
                      "title", "imdb_index", "phonetic_code", "note", "md5sum"],
        "company_name": ["id", "imdb_id", "name", "country_code", "name_pcode_nf", "name_pcode_sf", "md5sum"],
        "info_type": ["id", "info"],
        "char_name": ["imdb_id", "id", "name", "imdb_index", "name_pcode_nf", "surname_pcode", "md5sum"],
        "keyword": ["id", "keyword", "phonetic_code"],
        "movie_companies": ["movie_id", "company_id", "company_type_id", "id", "note"],
        "aka_name": ["person_id", "id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                     "md5sum"], "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "nr_order", "role_id", "note"]}

    try:
        cursor.execute("ROLLBACK")

        Parsed_Query = parse(view_script)
        #Getting the view tables
        Tables_list = Parsed_Query['from']

        #Getting the full name of the table
        left_table = Tables_list[0]["value"]
        right_table = Tables_list[1]["value"]

        #Getting the aliases of the tables
        left_table_alias = Tables_list[0]["name"]
        right_table_alias = Tables_list[1]["name"]

        #Concatenating the aliases with column names
        left = [left_table_alias + "." + element for element in schema[left_table]]
        right = [right_table_alias + "." + element for element in schema[right_table]]

        #Generating unique aliase  (table name + _ + table alias)
        left_aliases = [ left[i].split('.')[0]+"_"+left[i].split('.')[1]  for i in range(len(left))]
        left_projections = [element + ' AS ' + alias for element, alias in zip(left, left_aliases)]

        right_aliases = [right[i].split('.')[0]+"_"+right[i].split('.')[1]  for i in range(len(right))]
        right_projections = [element + ' AS ' + alias for element, alias in zip(right, right_aliases)]

        # Create the materialized view with unique column aliases
        try:
            create_view_statement = f"CREATE MATERIALIZED VIEW V_{view_name} AS SELECT   {', '.join(left_projections)} , {', '.join(right_projections)} {view_script[8:]}"
            cursor.execute(create_view_statement)
            print(f"The view 'V_{view_name}' has been materialized.")
        except psycopg2.errors.DiskFull as e:
            print("Disk full error:", e)
            print("Skipping to the next statement...")

        except Error as e:
            print("Error executing the SQL statement:", e)
    except ProgrammingError as e:
        print(f"An error occurred while materializing the view '{view_name}': {e}")
