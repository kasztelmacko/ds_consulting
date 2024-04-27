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
    There is a problem with saving August, so it is replaced with correct value
    '''
    data[date_column] = data[date_column].str.replace(r'(st|nd|rd|th|)', '', regex=True).str.replace('Augu', 'August')
    data[date_column] = pd.to_datetime(data[date_column], format='%d %B %Y')
    return data

def convert_Route(data):
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

def convert_AirlineName(data):
    '''
    Function to convert AirlineName as it stores same information as Slug
    '''
    data['AirlineName'] = data['AirlineName'].str.replace(' ', '-').replace("'","").str.lower()
    return data

def conver_Verified(data):
    '''
    Function to convert Verified Flighhts column to boolean
    '''
    data['TripVerified'] = data['TripVerified'].replace('NotVerified', 'Not Verified')
    data['TripVerified'] = data['TripVerified'].apply(lambda x: 'Not Verified' if x not in ['Trip Verified', 'Not Verified'] else x)
    data['TripVerified'] = data['TripVerified'].map({'Trip Verified': 1, 'Not Verified': 0})
    return data

def remove_overallscore_NaN(data):
    '''
    Function to remove rows with NaN values in OverallScore as the goal of the project is to determine what factors/individual ratings influence overall score
    '''
    data = data.dropna(subset=['OverallScore'])
    return data

def convert_Recommended(data):
    '''
    Function to convert Recommended column to boolean
    '''
    data['Recommended'] = data['Recommended'].map({'yes': 1, 'no': 0})
    return data

if __name__ == '__main__':
    data = load_data('data/AirlineReviews.csv')
    get_size(data)
    
    data = extract_month_year(data, 'DatePub')
    data = fill_Route(data)
    data = convert_Route(data)
    data = convert_AirlineName(data)
    data = conver_Verified(data)
    data = convert_Recommended(data)

    '''
    Remove columns that are not needed
    unique_id: no value added
    Route: replaced with Departure and Arrival
    Slug: same as AirlineName
    Aircraft: too much NaN values
    Title: no added value to Review
    DateFlown: Reviews are usually given after few days of the flight, and DatePub has no NaN values
    '''
    data = remove_columns(data, ['unique_id', 'Route', "Slug", "Aircraft", "Title", "DateFlown"])
    data = remove_overallscore_NaN(data)
    data.to_csv('data/AirlineReviews_cleaned.csv', index=False)
