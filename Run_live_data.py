import csv
from statistics import median

# Define the window size for the median filter
window_size = 149
# This file is used to run the life data to simulate the numbers coming into a real time surgery 
# To run, go to cmd window and type "cd C:\Users\dema2\OneDrive\Desktop\PhD\RFMN\Reflex-Fuzzy-Network"
# Then type python Run_live_data.py
import csv
import random
import time




def median_filter(data):
    filtered_data = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        filtered_data.append(median(window))
    return filtered_data


def generate_values():
    val1 = random.uniform(4.3, 7.9)  
    if 4.3 <= val1 <= 5.8:
        val2 = random.uniform(2.3, 4.4)
        val3 = random.uniform(1.0, 1.9)
        val4 = random.uniform(0.1, 0.6)
    elif 4.9 <= val1 <= 7.0:
        val2 = random.uniform(2.0, 3.4)
        val3 = random.uniform(3.0, 5.1)
        val4 = random.uniform(1.0, 1.8)
    elif 4.9 <= val1 <= 7.9:
        val2 = random.uniform(2.2, 3.8)
        val3 = random.uniform(4.5, 6.9)
        val4 = random.uniform(1.4, 2.5)
    else:
        val2 = None  
        val3 = None  
        val4 = None  
    return val1, val2, val3, val4

Time = 0
sepal_length, sepal_width, petal_length, petal_width = generate_values()

fieldnames = ["Time", "sepal-length", "sepal-width", "petal-length", "petal-width"]

with open('Run_live_data.csv', 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:

    with open('Run_live_data.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "Time": Time,
            "sepal-length":sepal_length,
            "sepal-width":sepal_width,
            "petal-length":petal_length,
            "petal-width":petal_width
        }

        csv_writer.writerow(info)
        print(Time, sepal_length, sepal_width, petal_length, petal_width)

        Time += 1
        sepal_length, sepal_width, petal_length, petal_width = generate_values()

    with open('Run_live_data.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)  

        header = csv_reader.fieldnames
        data = {col: [] for col in header}

        for row in csv_reader:
            for col in header:
                data[col].append(float(row[col]))
        filtered_data = {col: median_filter(values) for col, values in data.items()}

    with open('output_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        num_rows = len(filtered_data[header[0]])  
        for i in range(num_rows):
            csv_writer.writerow([filtered_data[col][i] for col in header])

    time.sleep(.0050)




    



