<<<<<<< HEAD
"""
Module: arduino_live_50.py
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file opens arduino_live.csv and collects the most recent (last most) 50 lines
            and saves them into an arduino_live_50.csv file for real-time analysis. 
"""

import time, csv

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

=======
"""
Module: arduino_live_50.py
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file opens arduino_live.csv and collects the most recent (last most) 50 lines
            and saves them into an arduino_live_50.csv file for real-time analysis. 
"""

import time, csv

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

>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
