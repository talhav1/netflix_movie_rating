import os
from datetime import datetime
import numpy as np

def convert_years(year):
    # Define the reference date
    reference_date = datetime(1997, 1, 1)

    # Create a date for January 1 of the given year
    current_date = datetime(year, 1, 1)

    # Calculate the difference in days
    days_difference = (current_date - reference_date).days
    return days_difference


def load_files(directory):
    # List all entries in the directory
    all_entries = os.listdir(directory)
    
    # Initialize dictionaries to store the data
    datasets = {}
    
    # Process files
    for file in all_entries:
        if file == 'movie_dates.txt':
            file_path = os.path.join(directory, file)
            with open(file_path) as f:
                data = [convert_years(int(row.split(',')[0])) for row in f]
                data = np.array(data)
                name = os.path.splitext(file)[0]
                datasets[name] = data
                continue
        file_path = os.path.join(directory, file)
        data = np.loadtxt(file_path)
        data[data == 0] = np.nan
        name = os.path.splitext(file)[0]
        datasets[name] = data
    
    return datasets