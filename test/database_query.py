import ROOT
import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
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

def save_dataframe(csv_name, file_name = "dataframe.root", tree_name = "tree"):
    print("Save to", f"{file_name}:{tree_name}" )
    # Write the dataframe to a ROOT file
    df = ROOT.RDF.MakeCsvDataFrame(csv_name)
    df.Snapshot(tree_name, file_name)

def csv_to_root_rdataframe(csv_file):
# Read the CSV file into a pandas DataFrame
    if os.path.isdir(csv_file):
        all_data = []
        for filename in os.listdir(csv_file):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_file, filename)
                data = pd.read_csv(file_path)
                all_data.append(data)
        #df2 = pd.concat(all_data, ignore_index=True)
        if len(all_data) == 0:
            exit()
        df2 = pd.concat(all_data, ignore_index=True)
        df2.to_csv("tmp.csv", index=False)
        # Create a RDataFrame from the CSV file
        df = ROOT.RDF.MakeCsvDataFrame("tmp.csv")
    else:
        if csv_file.endswith(".csv"):
            df = ROOT.RDF.MakeCsvDataFrame(csv_file)
        elif csv_file.endswith(".root"):
            df = ROOT.RDataFrame("tree", csv_file)
        else:
            raise TypeError("Unrecognised file syntax")

    # By default return the dataframe
    return df


def plot_variables_vs(df, selection, var1_name, var2_name, var1_error_name="", var2_error_name=""):
    # Apply the selection filter
    filtered_data = df.Filter(selection)
    
    # Parse var2_name if it's provided as a list in string format
    if var2_name.startswith("[") and var2_name.endswith("]"):
        var2_names = var2_name[1:-1].split(",")
        var2_names = [name.strip() for name in var2_names]
    else:
        var2_names = [var2_name]
    
    if var2_error_name.startswith("[") and var2_error_name.endswith("]"):
        var2_error_names = var2_error_name[1:-1].split(",")
        var2_error_names = [name.strip() for name in var2_error_names]
    else:
        var2_error_names = [var2_error_name] if var2_error_name else []

    # Extract data for variable1 and optionally for errors
    columns_to_extract = [var1_name] + var2_names
    if var1_error_name:
        columns_to_extract.append(var1_error_name)
    columns_to_extract += var2_error_names

    result = filtered_data.AsNumpy(columns=columns_to_extract)
    var1_data = np.array(result[var1_name])
    var1_error_data = np.array(result[var1_error_name]) if var1_error_name else None
    # Sort data by var1_data
    sort_indices = np.argsort(var1_data)
    var1_data = var1_data[sort_indices]
    if var1_error_data is not None:
        var1_error_data = var1_error_data[sort_indices]

    plt.figure(figsize=(10, 6))
    
    # Plot data for each variable in var2_names
    for idx, v in enumerate(var2_names):
        var2_data = np.array(result[v])[sort_indices]
        if var2_error_names:
            var2_error_data = np.array(result[var2_error_names[idx]])[sort_indices]
            plt.errorbar(var1_data, var2_data, xerr=var1_error_data, yerr=var2_error_data, label=v, fmt='o-', alpha=0.7, capsize=5, markersize=2)
        else:
            plt.plot(var1_data, var2_data, 'o-', label=v, alpha=0.5, markersize=5)

    plt.xlabel(var1_name)
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"{var1_name} vs Variables with selection: {selection}")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)  # This will not block the program execution

    # Ask for user input to close the plot
    user_input = input("Type 'q' or 'quit' to close the plot: ").strip().lower()
    if user_input.lower() == "q" or user_input.lower() == "quit":
        plt.close()

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
        filtered_df.Display(columns, 100000).Print()
    except Exception as e:
        print(f"Error encountered: {e}")

    return filtered_df
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: csv_to_ntuple.py <csv_file>")
        sys.exit(1)

    #ROOT.EnableImplicitMT() 
    csv_name = os.path.abspath(sys.argv[1])
    df = csv_to_root_rdataframe(csv_name)# sys.argv[2])
    column_names = df.GetColumnNames()
    #print("Available column names:\n")
    names = ""
    columns_origin = []
    for name in column_names:
        names += f"{name}\t"
        columns_origin.append(f'{name}')
    columns = deepcopy(columns_origin)
    #print(names)
    selections = "1"
    print(columns)
    re_print_help = True
    draw_plot = False
    # print("\033[1;91mBold Red Text\033[0m")
    # print("\033[1;92mBold Green Text\033[0m")
    # print("\033[1;93mBold Yellow Text\033[0m")
    # print("\033[1;94mBold Blue Text\033[0m")
    # print("\033[1;95mBold Magenta Text\033[0m")
    # print("\033[1;96mBold Cyan Text\033[0m")
    # print("\033[1;97mBold White Text\033[0m")
    while True:
        while True:
            query_help_front = "\033[1;92mEnter the query keyword:\033[0m"
            query_help = "\033[1;94m(allowed ones are:\033[0m\n"
            query_help += "* \033[1;93mselect, sel\033[0m      enter the selection string to filter the data\n"
            query_help += "* \033[1;93mchannel, ch\033[0m      enter the selection to pick a single channel, format: run,pos,ch\n"
            query_help += "* \033[1;93mtile, t\033[0m          enter the selection to pick a single tile, format: run,pos\n"
            query_help += "* \033[1;93mshowallcol\033[0m       show all column names\n"
            query_help += "* \033[1;93mcol\033[0m              enter the column names to show in the table\n"
            query_help += "* \033[1;93mdraw\033[0m             draw the results var1 vs var2\n"
            query_help += "* \033[1;93mok\033[0m               configuration finished and show the results\n"
            query_help += "* \033[1;93msave\033[0m             save the data in a root file\n"
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
            elif query.lower() == "channel" or  query.lower() == "ch":
                sel_str =  input("Enter the selection, format: run,pos,ch\n")
                run = sel_str.split(",")[0]
                pos = sel_str.split(",")[1]
                ch  = sel_str.split(",")[2]
                selections = f"run=={run} and pos=={pos} and ch=={ch}"
            elif query.lower() == "tile" or  query.lower() == "t":
                sel_str =  input("Enter the selection, format: run,pos\n")
                run = sel_str.split(",")[0]
                pos = sel_str.split(",")[1]
                selections = f"run=={run} and pos=={pos}"
            elif query.lower() == "col":
                # Get the column names from user input
                column_input = input("Enter the columns to show: col1,col2,col3,... \n")
                columns = [col.strip() for col in column_input.split(',')]
                all_match = True
                for test_name in columns:
                    name_match = False
                    for name in column_names:
                        if test_name == f'{name}':
                            name_match = True
                    if not name_match:
                        all_match = False
                        print("\033[1;91mPlease enter correct column names from available ones:\033[0m")
                if not all_match:
                    columns = deepcopy(columns_origin)
                print(columns)
            elif query.lower() == "showallcol":
                print(names)
            elif query.lower() == "save":
                tree_name = input("\033[1;93mEnter the name for the TTree object, default is\033[0m \033[1;93mtree\033[0m\n")
                file_name = input("\033[1;93mEnter the name for the TFile, default is\033[0m \033[1;93mdataframe.root\033[0m\n")
                #tree_name = tree_name.strip()
                if tree_name == "":
                    tree_name = "tree"
                save_dataframe(csv_name, file_name, tree_name)
            elif query.lower() == "draw":
                variables = input("\033[1;93mEnter the variables to draw, allowed format:\n varY varX \n varY varX varY_err\n varY varX varY_err varX_err\033[0m\n")    
                variable_components = variables.split(" ")
                if len(variable_components) == 2:
                    varY = variables.split(" ")[0]
                    varX = variables.split(" ")[1]
                    varY_err = "" #variables.split(",")[1]
                    varX_err = "" #variables.split(",")[1]
                elif len(variable_components) == 3:
                    varY = variables.split(" ")[0]
                    varX = variables.split(" ")[1]
                    varY_err = variables.split(" ")[2]
                    varX_err = "" #variables.split(",")[1]
                elif len(variable_components) == 4:
                    varY = variables.split(" ")[0]
                    varX = variables.split(" ")[1]
                    varY_err = variables.split(" ")[2]
                    varX_err = variables.split(" ")[3]
                else:
                    print("please enter correct formats: var_Y,var_X or var_Y,var_Y_err,var_X")
                    
                draw_plot = True
            elif query.lower() == "quit" or query.lower() == "q":
                ROOT.gSystem.Exit(0)
                exit()
            elif query.lower() == "ok":
                break
            else:
                print("\033[91mplease enter the correct keyword for the search\033[0m\n")
                print(query_help)
            re_print_help = False
        if draw_plot:
            plot_variables_vs(df, selections, varX, varY, varX_err, varY_err)
        else:
            filtered_df = print_filtered_data(df, selections, columns)
    





