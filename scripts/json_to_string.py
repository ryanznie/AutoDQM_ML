import argparse
import json

# Create an argument parser
parser = argparse.ArgumentParser(description="Convert JSON data to a comma-separated string.")
parser.add_argument('-i', '--input', type=str, help='Path to the input JSON file')

# Parse the command-line arguments
args = parser.parse_args()

# Check if the input argument is provided
if args.input:
    input_path = args.input
else:
    print("Please provide the input JSON file using the -i flag.")
    exit()

# Load the JSON data from the file
with open(input_path, 'r') as file:
    json_data = json.load(file)

# Extract the list from the JSON data
data_list = json_data["L1T"]
data_header = list(json_data.keys())[0]

# Convert the list into a comma-separated string
csv_string = ",".join([data_header + "/" + value for value in data_list])

# Print the comma-separated string
print(csv_string)
