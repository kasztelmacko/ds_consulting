import numpy as np
import pandas as pd
import re


def load_data(path):
    '''
    Function to load data from a csv file
    '''
    data = pd.read_csv(path)
    return data

def get_size(data):
    '''
    Function to get the size of the dataset
    '''
    return print(f"dataset has {data.shape[0]} rows and {data.shape[1]} columns")

def remove_columns(data, columns):
    '''
    Function to remove unnecesery columns from a dataset
    '''
    data = data.drop(columns=columns)
    return data

def extract_month_year(data, date_column):
    '''
    Function to extract month and year from a date column, and convert it to a numeric value
    '''
    data[date_column] = data[date_column].str.replace(r'(st|nd|rd|th)', '')
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data[date_column] = data[date_column].dt.strftime('%B %Y')
    return data

def normalize_column_Route(data):
    '''
    Function to normalize a column in a dataset (change Route to departure and arrival)
    '''
    data["Route"] = data["Route"].replace("  ", " ")
    data[['Departure', 'Arrival']] = data["Route"].str.split(' to ', expand=True)
    return data

def fill_Route(data):
    '''
    Function to fill missing values in a Route column
    '''
    pattern = r'(\w+ to \w+|\w+-\w+|\w+ - \w+)' # pattern to extract the route
    data['Route'] = data['Review'].str.extract(pattern, expand=False).str.replace('-', ' to ')
    data['Route'] = data['Route'].fillna('Unknown')
    return data

if __name__ == '__main__':
    data = load_data('data/AirlineReviews.csv')
    get_size(data)
    data = extract_month_year(data, 'DatePub')
    data = fill_Route(data)
    data = normalize_column_Route(data)
    data = remove_columns(data, ['unique_id', 'Route'])
    print(data.head(5))