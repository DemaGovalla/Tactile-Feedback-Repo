import time
import csv


file_name = 'Arduino_live.csv'
output_file = 'output_file.csv'
# def read_last_50(file_name, output_file):
while True:
    with open(file_name, 'r') as file:
        lines = file.readlines()
        last_50 = lines[-50:]  # Get the last 50 lines or data points
        
        # Process the last 50 data points (lines)
        # Here, you might perform operations or store these points as needed
        # For demonstration, printing the last 50 lines
        for line in last_50:
            print(line.strip())  # Stripping to remove newline characters
        
        # Write the last 50 lines to another CSV file
        with open(output_file, 'w', newline='') as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerows([line.strip().split(',') for line in last_50])
        
    time.sleep(1/50)  # Refresh rate of 50 Hz (0.02 seconds delay)

# Replace 'input_file.csv' and 'output_file.csv' with actual file names
# read_last_50('Arduino_live.csv', 'output_file.csv')