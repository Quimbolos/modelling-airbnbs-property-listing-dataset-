# %%
def clean_tabular_data(dataframe):

    def remove_rows_with_missing_ratings(dataframe):
        dataframe = dataframe.dropna(subset=['Cleanliness_rating',
        'Accuracy_rating', 'Communication_rating', 'Location_rating',
        'Check-in_rating', 'Value_rating'])

        return dataframe

    def combine_description_strings(dataframe):
        for i in range(len(dataframe['Description'])):

            if type(dataframe['Description'][i]) == str:
                string = dataframe['Description'][i].strip('][').split(', ')
                newstring = ''

                for j in range(len(string)):
                    output = string[j].replace("'","")
                    output = output.replace(" l "," l'")
                    output = output.replace(" d "," d'")
                    output = output.replace("\\n\\n","")
                    output = output.replace("\\n",".")
                    output = output.replace("..",".")
                    if output == 'About this space':
                        newstring = newstring
                        continue
                    elif output == '"The space':
                        newstring = newstring
                        continue
                    elif output == 'Guest access':
                        newstring = newstring
                        continue
                    elif output == 'Sanitary Facilities':
                        newstring = newstring
                        continue
                    elif output == 'Other things to note':
                        newstring = newstring
                        continue
                    elif j > 0:
                        newstring = newstring + output

                dataframe['Description'][i] = newstring
        
        dataframe['Description'].astype(str).str[0].replace("\"","") # Not Working

        return dataframe

    def set_default_feature_values(dataframe):
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
    
