<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""


import csv, random, time

def generate_random_data():
    return {
        'leave': random.randint(1, 4),
        'set': random.randint(1, 4),
        'house': random.randint(1, 4),
        'farm': random.randint(1, 4),
        'phone': random.randint(1, 4),
        'keys': random.randint(1, 4)
    }

def write_to_csv(file_path, data, write_header=False):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['leave', 'set', 'house', 'farm', 'phone', 'keys']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            csv_writer.writeheader()

        if data:
            csv_writer.writerow(data)

def clear_csv(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        pass

def update_csv_periodically(file_path, interval_seconds):
    clear_csv(file_path)  # Clear the CSV initially
    write_to_csv(file_path, None, write_header=True)  # Write header initially

    while True:
        random_data = generate_random_data()
        write_to_csv(file_path, random_data)
        print(f"Updated CSV: {random_data}")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    csv_file_path = "TactileFeedback_data.csv"
    update_interval_seconds = .02

    update_csv_periodically(csv_file_path, update_interval_seconds)
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""


import csv, random, time

def generate_random_data():
    return {
        'leave': random.randint(1, 4),
        'set': random.randint(1, 4),
        'house': random.randint(1, 4),
        'farm': random.randint(1, 4),
        'phone': random.randint(1, 4),
        'keys': random.randint(1, 4)
    }

def write_to_csv(file_path, data, write_header=False):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['leave', 'set', 'house', 'farm', 'phone', 'keys']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            csv_writer.writeheader()

        if data:
            csv_writer.writerow(data)

def clear_csv(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        pass

def update_csv_periodically(file_path, interval_seconds):
    clear_csv(file_path)  # Clear the CSV initially
    write_to_csv(file_path, None, write_header=True)  # Write header initially

    while True:
        random_data = generate_random_data()
        write_to_csv(file_path, random_data)
        print(f"Updated CSV: {random_data}")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    csv_file_path = "TactileFeedback_data.csv"
    update_interval_seconds = .02

    update_csv_periodically(csv_file_path, update_interval_seconds)
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
