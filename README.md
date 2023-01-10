# Modelling Airbnb's property listing dataset 

> This project builds a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Language and tools

<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>  <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> 
</a>  <a href="https://aws.amazon.com/?nc2=h_lg"><img src="https://d0.awsstatic.com/logos/powered-by-aws-white.png" alt="Powered by AWS Cloud Computing" width="110" height="40"/> </a> </p>

## Milestone 1: Set up the environment

> Create a new GitHub repo to upload the code and allow version control throughout the project.

## Milestone 2: Get an overview of the system you're about to build

> Get an overview of the framework for evaluating a wide range of machine learning models that can be applied to varius datasets

#### Building a Modelling Framework Video:

<p align="left"> <a href="https://www.youtube.com/watch?v=Tub0xAsNzk8"> <img src=https://global-uploads.webflow.com/62cf31426220d61ad2c45936/62d42da3aa648fd85d1313d0_logo-large.webp width="150" height="40"/> </a></p> 

## Milestone 3: Data preparation

> Before building the framework, the Airbnb dataset is structured and cleaned

Initially, the tabular dataset is downloaded from this [link](https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip).

The tabular dataset has the following columns:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
- Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

In Task 1 a file named ```tabular_data.py``` is created:

- As there are missing values in the rating columns, a function called ```remove_rows_with_missing_ratings``` is created which removes the rows with missing values in these columns. It takes in the dataset as a pandas dataframe and returns it as the same type.

```python
def remove_rows_with_missing_ratings(dataframe):
        
    dataframe = dataframe.dropna(subset=['Cleanliness_rating',
    'Accuracy_rating', 'Communication_rating', 'Location_rating',
    'Check-in_rating', 'Value_rating'])
    dataframe.drop(['Unnamed: 19'], axis=1, inplace = True)

    return dataframe

```

 - In addition, the "Description" column contains lists of strings. A function called ```combine_description_strings``` is created to combine the list items into the same string. 

 ```python
def combine_description_strings(dataframe):

    dataframe['Description'] = dataframe['Description'].astype(str)
    dataframe['Description'] = dataframe['Description'].str.replace('About this space', '')
    dataframe['Description'] = dataframe['Description'].str.replace("'", '')
    dataframe['Description'] = dataframe['Description'].apply(lambda x: x.strip())
    dataframe['Description'] = dataframe['Description'].str.split(",")
    dataframe['Description'] = dataframe['Description'].apply(lambda x: [i for i in x if i != ''])
    dataframe['Description'] = dataframe['Description'].apply(lambda x: ''.join(x)).astype(str)
    dataframe['Description'] = dataframe['Description'].str.replace("'", '')
    dataframe['Description'] = dataframe['Description'].str.replace("The space The space\\\\n", 'The space: ')
    dataframe['Description'] = dataframe['Description'].str.replace("Guest access", 'Guest access: ')
    dataframe['Description'] = dataframe['Description'].str.replace("Sanitary facilities", 'Sanitary Facilities: ')
    dataframe['Description'] = dataframe['Description'].str.replace("Other things to note\\\\n", 'Other things to note: ')
    dataframe['Description'] = dataframe['Description'].str.replace("\\\\n\\\\n", '. ')
    dataframe['Description'] = dataframe['Description'].str.replace("\\\\n", ' ')
    dataframe['Description'] = dataframe['Description'].str.replace("\.\.", '.')


    return dataframe

```

- Furthermore, the "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. To deal with this, a function called ```set_default_feature_values``` replaces these entries with the number 1.

```python
def set_default_feature_values(dataframe):
        
    dataframe[['guests', 'beds','bathrooms', 'bedrooms']] = dataframe[['guests', 'beds','bathrooms', 'bedrooms']].fillna(value=1)

    return dataframe

```

Finally, all of the code that does this processing is included into a function called ```clean_tabular_data``` which takes in the raw dataframe, calls these functions sequentially on the output of the previous one, and returns the processed data.