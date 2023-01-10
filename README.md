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
- Check-in_rate: The rating of the check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

In Task 1 a file named ```tabular_data.py``` is created:

- As there are missing values in the rating columns, a function called ```remove_rows_with_missing_ratings``` is created. This function removes the rows with missing values in certain columns.

```python
def remove_rows_with_missing_ratings(dataframe):
        
    dataframe = dataframe.dropna(subset=['Cleanliness_rating',
    'Accuracy_rating', 'Communication_rating', 'Location_rating',
    'Check-in_rating', 'Value_rating'])
    dataframe.drop(['Unnamed: 19'], axis=1, inplace = True)

    return dataframe

```

 - The "Description" column contains lists of strings. A function called ```combine_description_strings``` is created to combine the list items into the same string. 

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

- The "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. To deal with this without dropping these rows, a function called ```set_default_feature_values``` replaces the null entries with the number 1.

```python
def set_default_feature_values(dataframe):
        
    dataframe[['guests', 'beds','bathrooms', 'bedrooms']] = dataframe[['guests', 'beds','bathrooms', 'bedrooms']].fillna(value=1)

    return dataframe

```

- All the code that does this processing is included in a function called ```clean_tabular_data```, which takes in the raw dataframe, calls these functions sequentially on the output of the previous one, and returns the processed data.

```python
def clean_tabular_data():

    def remove_rows_with_missing_ratings(dataframe):
        # CODE
        return dataframe

    def combine_description_strings(dataframe):
        # CODE
        return dataframe

    def set_default_feature_values(dataframe):
        # CODE
        return dataframe

    dataframe = combine_description_strings(dataframe)
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = set_default_feature_values(dataframe)

    return dataframe 

```

- Finally, inside an if ```__name__ == "__main__"``` block, the ```tabular_data.py``` main code logic is as follows:

  - Load the raw dataset using pandas
  - Call ```clean_tabular_data``` on it
  - Save the processed data as ```clean_tabular_data.csv``` in the same folder where the raw tabular data is.

```python
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

```

In Task 2 a file named ```prepare_image_data.py``` is created:

- Firstly, all the Airbnb's images inside the .zip file downloaded before are uploaded into an [aws bucket](https://s3.console.aws.amazon.com/s3/buckets/myfirstbucketjbf?region=eu-west-2&tab=objects), using the Amazon S3 service.

- Secondly, the function called ```download_images``` is created. This function downloads the image folder for each apartment from S3 and puts them into a folder called images. 

```python
def download_images():

    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket('myfirstbucketjbf') 
    for obj in bucket.objects.filter(Prefix = 'images'):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key) # save to same path

```
- Thirdly, a function called ```resize_images``` is created. This function loads each RGB formatted image and resizes it to the same height and width before saving the new version in the newly created ```processed_images``` folder.
The height of the resized images is set to be the smallest image height in the dataset. In addition, when resizing the image, the code maintains the aspect ratio of the image and adjusts the width proportionally to the change in height.

```python
def resize_images():
      
    base_dir = os.path.join(os.getcwd(),"images")

    rgb_file_paths = []

    # Get the RGB images file paths
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, f)
                if os.path.isfile(file_path):
                    with Image.open(file_path) as img:
                        if img.mode == 'RGB':
                            rgb_file_paths.append(file_path)
    
    # Set the height of the smallest image as the height for all of the other images
    min_height = float('inf')
    for checked_file in rgb_file_paths:
        with Image.open(checked_file) as im:
            min_height = min(min_height, im.height)

    # Create the processed_images folder
    processed_images_path = os.path.join(os.getcwd(),"processed_images")
    if os.path.exists(processed_images_path) == False:
        os.makedirs(processed_images_path)

    # Resize all images to have the min_height whilst mantaining the aspect ratio
    for file_path in rgb_file_paths:
        with Image.open(file_path) as im:
            width, height = im.size
            new_height = min_height
            new_width = int(width * new_height / height)
            resized_im = im.resize((new_width, new_height))
            resized_im.save(os.path.join('processed_images', os.path.basename(file_path)))

        return

```

- Finally, as it has been done in the previous task, all the functions above are placed within the ```process_images``` function, which calls these functions sequentially. Then, this function is called inside an if ```__name__ == "__main__"``` block.

```python
def process_images():

    def download_images():
        #CODE
        return

    def resize_images():
        #CODE
        return

    download_images()
    resize_images()

    return


if __name__ == "__main__":
    process_images()

```