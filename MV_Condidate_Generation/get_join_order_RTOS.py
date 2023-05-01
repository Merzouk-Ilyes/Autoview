import re
def Get_JO_from_file():

    with open('myfile.sql', 'r') as file:
        sql_content = file.read()

    match = re.search(r'/\*+(.*?)\*/', sql_content, re.DOTALL)

    if match:
        header = match.group(1)
        # Find the text after "Leading"
        leading_match = re.search(r'Leading\s*\(\s*(.*?)\s*\)', header)
        if leading_match:
            leading_text = leading_match.group(1)
            # Remove parentheses
            leading_text = leading_text.replace('(', '').replace(')', '')
            # Split into an array
            leading_array = leading_text.split()
            print(leading_array)
        else:
            print("Leading text not found in header.")
    else:
        print("Header not found.")
