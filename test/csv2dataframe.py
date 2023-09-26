import ROOT

def column_to_list(rdf, column_name):
    """
    Convert a specific column of an RDataFrame to a Python list.
    
    Parameters:
    - rdf: The input RDataFrame.
    - column_name: Name of the column to be converted.
    
    Returns:
    - A list containing all the values of the specified column.
    """
    
    return list(rdf.AsNumpy([column_name])[column_name])

def csv_to_root_rdataframe(csv_file, root_file):
    # Create a RDataFrame from the CSV file
    df = ROOT.RDF.MakeCsvDataFrame(csv_file)

    # Write the dataframe to a ROOT file
    df.Snapshot("tree", root_file)

    # By default return the dataframe
    return df

def print_filtered_data(df, filter_string, columns):
    """
    Display specified columns of an RDataFrame.
    
    Parameters:
    - rdf: The input RDataFrame.
    - columns: List of columns to display.
    - n_rows (optional): Number of rows to display. Default is 10.
    
    Returns:
    None.
    """
        
    filtered_df = df.Filter(filter_string)
    column_names = df.GetColumnNames()
    for test_name in columns:
        name_match = False
        for name in column_names:
            if test_name == f'{name}':
                name_match = True
        if not name_match:
            print("\033[1;91mPlease enter correct column names from available ones:\033[0m")
            print(column_names)
            return df
    try:
        filtered_df = df.Filter(filter_string)
        #filtered_df.Count()  # Triggering execution
        # Print the filtered rows
        #filtered_df.Display(20).Print()
        filtered_df.Display(columns, 100000, 100000).Print()
    except Exception as e:
        print(f"Error encountered: {e}")

    return filtered_df
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: csv_to_ntuple.py <csv_file>")
        sys.exit(1)

    df = csv_to_root_rdataframe(sys.argv[1],"df.root")# sys.argv[2])
    column_names = df.GetColumnNames()
    #print("Available column names:\n")
    names = ""
    columns = []
    for name in column_names:
        names += f"{name}\t"
        columns.append(f'{name}')
    #print(names)
    selections = "1"
    print(columns)
    re_print_help = True
    # print("\033[1;91mBold Red Text\033[0m")
    # print("\033[1;92mBold Green Text\033[0m")
    # print("\033[1;93mBold Yellow Text\033[0m")
    # print("\033[1;94mBold Blue Text\033[0m")
    # print("\033[1;95mBold Magenta Text\033[0m")
    # print("\033[1;96mBold Cyan Text\033[0m")
    # print("\033[1;97mBold White Text\033[0m")

    while True:
        query_help_front = "\033[1;92mEnter the query keyword:\033[0m"
        query_help = "\033[1;94m(allowed ones are:\033[0m\n"
        query_help += "* \033[1;93mSelect\033[0m           enter the selection string to filter the data\n"
        query_help += "* \033[1;93mShowAllCol\033[0m       show all column names\n"
        query_help += "* \033[1;93mCol\033[0m              enter the column names to show in the table\n"
        query_help += "* \033[1;93mOK\033[0m               configuration finished and show the results\n"
        query_help += "* \033[1;93mquit , q\033[0m         quit the query\n"
        query_help += "\033[1;96m(Uppercase and lowercase are not distinguished)\033[0m"
        if re_print_help:
            print(query_help_front)
            print(query_help)
        else:
            print(query_help_front)
        query = input().strip()
        if query.lower() == "select":
            selections =  input("Enter the selection:\n")
        elif query.lower() == "col":
            # Get the column names from user input
            column_input = input("Enter the columns to show: col1,col2,col3,... \n")
            columns = [col.strip() for col in column_input.split(',')]
            print(columns)
        elif query.lower() == "showallcol":
            print(names)
        elif query.lower() == "quit" or query.lower() == "q":
            exit()
        elif query.lower() == "ok":
            break
        else:
            print("\033[91mplease enter the correct keyword for the search\033[0m\n")
            print(query_help)
        re_print_help = False

    filtered_df = print_filtered_data(df, selections, columns)
    






