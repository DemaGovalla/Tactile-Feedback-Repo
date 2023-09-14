# This file is used to run the life data to simulate the numbers coming into a real time surgery 
# To run, go to cmd window and type "cd C:\Users\dema2\OneDrive\Desktop\PhD\RFMN\Reflex-Fuzzy-Network"
# Then type python Run_live_data.py
import csv
import random
import time
from datetime import datetime



# x = []
# x_value = int(datetime.now().strftime("%S"))
        # x.append(now.strftime("%H:%M:%S:%f"))

def generate_values():
    val1 = random.uniform(4.3, 7.9)  # Generate a random value for val1 between 0 and 10
    
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
        val2 = None  # You can assign a default value if val1 doesn't fall into any range
    
    return val1, val2, val3, val4

x_value = 0
total_1 = 0
total_2 = 0
total_3 = 0
total_4 = 0

fieldnames = ["x_value", "total_1", "total_2", "total_3", "total_4"]

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:

    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "x_value": x_value,
            "total_1":total_1,
            "total_2":total_2,

            "total_3":total_3,
            "total_4":total_4
        }

        csv_writer.writerow(info)
        print(x_value, total_1, total_2, total_3, total_4)

        x_value += 1
        # total_1 = random.random()
        # total_2 = random.random()

        total_1, total_2, total_3, total_4 = generate_values()

        # total_1 = random.uniform(0, 1)
        # total_2 = random.uniform(0, 1)

        # total_3 = random.uniform(0, 1)
        # total_4 = random.uniform(0, 1)

    time.sleep(.050)




    
