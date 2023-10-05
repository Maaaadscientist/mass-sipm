import ROOT
import os
import sys
import pandas as pd
from copy import deepcopy

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
    #print("Save to", f"{file_name}:{tree_name}" )
    # Write the dataframe to a ROOT file
    try:
        df = ROOT.RDF.MakeCsvDataFrame(csv_name)
        df.Snapshot(tree_name, file_name)
    except Exception as e:
        harvest_path = "/".join(csv_name.split("/")[:-1])
        with open(f"{harvest_path}/errors.log", "a+"):
            file.write(csv_name + "\n")
        print(f"An error occurred while saving to {file_name}. Error: {e}")

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

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: csv_to_ntuple.py <csv_file> <root_file>")
        sys.exit(1)
    csv_name = os.path.abspath(sys.argv[1])
    root_path =  os.path.abspath(sys.argv[2])
    output_dir = "/".join(root_path.split("/")[:-1])
    #print(output_dir)
    if not os.path.isdir(output_dir):
        #print("make dir:", output_dir)
        os.makedirs(output_dir)

    ROOT.EnableImplicitMT() 
    save_dataframe(csv_name, root_path)





