import csv
import sys

def is_float_or_int(value):
    """Check if a value is a float or int"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def check_csv_file(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Get header to determine the number of columns
        header = next(reader)
        num_columns = len(header)
        
        # Iterate over each row
        for row_num, row in enumerate(reader, start=1):
            # Check for continuous commas
            if ',,' in ','.join(row):
                print(f"Row {row_num} contains continuous commas.")
                continue

            # Check if the values are float or int numbers (excluding the first two columns)
            if not all(is_float_or_int(cell) for cell in row[2:]):
                print(f"Row {row_num} contains non-float/non-int values.")
                continue

if __name__ == "__main__":
    filename = sys.argv[1]
    check_csv_file(filename)

