def get_view_row_size(view_script, connection):
    # Create a cursor object
    cursor = connection.cursor()

    # Execute the query
    cursor.execute(view_script)
    # Fetch a single row from the result set
    row = cursor.fetchone()

    if row:
        # Calculate the storage size for each column in the row
        column_sizes = [
            len(str(value).encode())
            for value in row
        ]

        # Calculate the total storage size of the row
        row_size = sum(column_sizes)



    else:
        row_size = 0

    # Close the cursor and connection
    cursor.close()

    return row_size
