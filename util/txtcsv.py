import csv

# Define the input text file and output CSV file paths
input_file = '../data/matching_files.txt'
output_file = 'file_types.csv'

# Read the input text file and split the lines into lists of values
with open(input_file, 'r') as txtfile:
    lines = txtfile.readlines()
    data = [line.strip().split() for line in lines]

# Write the data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print(f"Converted {input_file} to {output_file}.")
