# Disaster Response Pipeline

## Introduction
This project is for categorizing 36 disaster classes(e.g. flood, strom, shelter, ...) based on the messages delivered during disasters. The training set of data is provided by [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. The project has been developed as part of Udacity Nano degree for data scientist.

The project contains:
1. ETL pipeline to wrangle data and load in SQLlight database.
2. Machine Learning pipeline to train classifier and optimize the classifier hyperparameters using GridSearchCV
3. Web application to visualize train data and show result of trained classifier

## File description
data:
  - data/proces_data.py: python script to clean data to save in SQLlite database
  - data/disaster_categories.csv and data/disaster_messages.csv: csv files with train data 
  
model:
  - model/train_classifier.py: python script to train classifier using machine learning algorithms and sklearn pipeline
  
app:
  - app/run.py: python script contains api with data visualization using plotly
  - templates/: directory that contains html for each api

## Usage
```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
  - Read the data from csv files to run ETL pipeline to create DisasterResponse.db SQLlite database
  
```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
  - Read the data from the database to train classifier
  
```python app/run.py``` 
  - Run web application
