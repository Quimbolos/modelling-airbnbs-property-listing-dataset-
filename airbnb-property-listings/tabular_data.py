# %%
def clean_tabular_data(dataframe):

    def remove_rows_with_missing_ratings(dataframe):
        '''
        Removes the rows with missing values in the columns specified

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame

        Returns
        -------
        dataframe: pandas.core.frame.DataFrame
            A pandas DataFrame without the rows with missing values
        
        '''
        dataframe = dataframe.dropna(subset=['Cleanliness_rating',
        'Accuracy_rating', 'Communication_rating', 'Location_rating',
        'Check-in_rating', 'Value_rating'])

        return dataframe

    def combine_description_strings(dataframe):
        '''
        Combines the list of strings from the Description Column into a single string for each row

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame

        Returns
        -------
        dataframe: pandas.core.frame.DataFrame
            A pandas DataFrame with a single string in the Description column
        '''

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

    def set_default_feature_values(dataframe):
        '''
        Replaces the empty values of certain coulumns with the number one

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame

        Returns
        -------
        dataframe: pandas.core.frame.DataFrame
            A pandas DataFrame no empty "guests", "beds", "bathrooms", and "bedrooms" entries
        '''
        dataframe[['guests', 'beds','bathrooms', 'bedrooms']] = dataframe[['guests', 'beds','bathrooms', 'bedrooms']].fillna(value=1)

        return dataframe

    dataframe = combine_description_strings(dataframe)
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = set_default_feature_values(dataframe)


    return dataframe 


import pandas as pd
import os


if __name__ == "__main__":

    # Load the raw data using pandas
    current_directory = os.getcwd()
    csv_relative_directory = 'airbnb-property-listings/tabular_data/listing.csv'
    csv_directory = os.path.join(current_directory, csv_relative_directory)
    df = pd.read_csv(csv_directory)

    # Call clean_tabular_data to process the dataframe
    df = clean_tabular_data(df)

    # Save the processed data as clean_tabular_data.csv in the same folder as the raw tabular data.
    df_realtive_directory = 'airbnb-property-listings/tabular_data/clean_tabular_data.csv'
    clean_tabular_data_directory = os.path.join(current_directory,df_realtive_directory)
    df.to_csv(clean_tabular_data_directory)

# %%
