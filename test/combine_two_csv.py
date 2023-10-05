import os
import sys
import pandas as pd


def read_csv_file(csv_file):
    if os.path.isdir(csv_file):
        all_data = []
        for filename in os.listdir(csv_file):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_file, filename)
                data = pd.read_csv(file_path)
                all_data.append(data)
        if len(all_data) == 0:
            raise ValueError(f"the {csv_file} has 0 rows")
        df = pd.concat(all_data, ignore_index=True)
    else:
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file)
        else:
            raise TypeError("invalid csv file type")
    return df

def fill_empty_elements(csv_str):
    # Replace consecutive commas in the middle
    while ',,' in csv_str:
        csv_str = csv_str.replace(',,', ',0,')
    
    # Add "0" to lines that end with a comma
    csv_str = "\n".join([line if not line.endswith(",") else line + "0" for line in csv_str.split("\n")])
    
    return csv_str


def main():
    csv_file1 = os.path.abspath(sys.argv[1])
    csv_file2 = os.path.abspath(sys.argv[2])
    output_path = os.path.abspath(sys.argv[3])
    if len(sys.argv) == 5:
        columns = sys.argv[4]
    else:
        # print("no columns names specified, use default keys: run, pos, ch, vol")
        columns = "run,pos,ch,vol"
    column_list = columns.split(",")
    df1 = read_csv_file(csv_file1)
    df2 = read_csv_file(csv_file2)
    # Merging the dataframes on multiple columns
    merged_df = pd.merge(df1, df2, on=column_list)
    
    # Convert DataFrame to CSV string and fill empty elements
    csv_str = merged_df.to_csv(index=False)
    modified_csv_str = fill_empty_elements(csv_str)
    
    output_dir = "/".join(output_path.split("/")[:-1])
    #print(output_dir)
    if not os.path.isdir(output_dir):
        # print("make dir:", output_dir)
        os.makedirs(output_dir)
    # Write to the output file
    with open(output_path, 'w') as f:
        f.write(modified_csv_str)


if __name__ == '__main__':
    main()

