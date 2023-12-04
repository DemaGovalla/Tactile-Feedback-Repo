import time
import csv

file_name = 'arduino_live.csv'
output_file = 'arduino_live_50.csv'
while True:
    with open(file_name, 'r') as file:
        lines = file.readlines()
        last_50 = lines[-50:]  # Get the last 50 lines or data points
        
        for line in last_50:
            print(line.strip())  # Stripping to remove newline characters
        
        # Write the last 50 lines to another CSV file
        with open(output_file, 'w', newline='') as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerows([line.strip().split(',') for line in last_50])
        
    time.sleep(1/50)  # Refresh rate of 50 Hz (0.02 seconds delay)

