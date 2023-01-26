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

In **Task 1** a file named ```tabular_data.py``` is created:

- As there are missing values in the rating columns, a function called ```remove_rows_with_missing_ratings()``` is created. This function removes the rows with missing values in certain columns.

```python
def remove_rows_with_missing_ratings(dataframe):
        
    dataframe = dataframe.dropna(subset=['Cleanliness_rating',
    'Accuracy_rating', 'Communication_rating', 'Location_rating',
    'Check-in_rating', 'Value_rating'])
    dataframe.drop(['Unnamed: 19'], axis=1, inplace = True)

    return dataframe

```

 - The "Description" column contains lists of strings. A function called ```combine_description_strings()``` is created to combine the list items into the same string. 

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

- The "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. To deal with this without dropping these rows, a function called ```set_default_feature_values()``` replaces the null entries with the number 1.

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
  - Call ```clean_tabular_data()``` on it
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

In **Task 2** a file named ```prepare_image_data.py``` is created:

- Firstly, all the Airbnb's images inside the .zip file downloaded before are uploaded into an [aws bucket](https://s3.console.aws.amazon.com/s3/buckets/myfirstbucketjbf?region=eu-west-2&tab=objects), using the Amazon S3 service.

- Secondly, the function called ```download_images()``` is created. This function downloads the image folder for each apartment from S3 and puts them into a folder called images. 

```python
def download_images():

    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket('myfirstbucketjbf') 
    for obj in bucket.objects.filter(Prefix = 'images'):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key) # save to same path

```
- Thirdly, a function called ```resize_images()``` is created. This function loads each RGB formatted image and resizes it to the same height and width before saving the new version in the newly created ```processed_images``` folder.
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

- Finally, as it has been done in the previous task, all the functions above are placed within the ```process_images()``` function, which calls these functions sequentially. Then, this function is called inside an if ```__name__ == "__main__"``` block.

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

In **Task 3**, within the ```tabular_data.py``` file, a function called ```load_airbnb()``` is created:

- This function returns the features and labels of your data in a tuple like (features, labels). The features are the variables that will allow us to predict the labels/target. In this case, the target to predict is the Price per Night. And all the other variables are the features from which we will try to predict the price per night. Text or image format features are not included yet; for now, only numerical tabular data.

```python
def load_airbnb():
    
    current_directory = os.getcwd()
    csv_relative_directory = 'airbnb-property-listings/tabular_data/clean_tabular_data.csv'
    csv_directory = os.path.join(current_directory, csv_relative_directory)
    df = pd.read_csv(csv_directory)
    labels = df['Price_Night']
    features = df.drop(['ID','Category','Title','Description','Amenities','Location','Price_Night','url','Unnamed: 0'], axis=1)
    
    return features, labels

```

## Milestone 4: Create a regression model
> Machine Learning models are created and evaluated. These models predict the price of the listing per night.

In **Task 1**, a file named ```modelling.py``` is created:

- Initially, the ```load_airbnb()``` function defined earlier is used to load in the dataset features ```X``` and the price per night (```Price_Night```) as the label ```y```. 

- Secondly, use ```StandardScaler()``` to standardise the features ```X```.

- Then, using the sklearn ```SGDRegressor()``` the ```Price_Night``` target is predicted from the features. 

- The dataset is split into training, validation and testing datasets:
    - The training data is used to train the model
    - The validation set should be used to help us choose the model. If we were to choose different models or configurations.
    - The testing set is used to evaluate the model's performance on unseen data.

- Seeding: 
    
    In the code below, note the following line: ```np.random.seed(10)```.

    - #### _Pseudo-random number generators_:
    
        Many ML algorithms employ random initialisation to, for example, instantiate the parameters of a linear regression model. Depending on the algorithm, it may have a more or less-severe effect on the result.

        - Each time you run an algorithm randomly, the result may vary to some degree.
        - Random number generators employ a ```seed```, a numerical value that determines what values will be generated.
        - For each run to be the same, (or to exhibit some phenomena similar to the case above), we should always seed all functions using random numbers.

            *The last one is quite easy in numpy and sklearn as it is a single line. Seeding via this approach is common in most frameworks*.
    - #### _Benefits of seed initialisation_:
        - To ensure the reproducibility of experiments, which is particularly important in ML.
        - To ensure an equal outcome for all runs.
        - Always set a random seed to ensure your results are repeatable when some part of the code involves random numbers being generated.

When splitting the data, if ```np.random.seed(10)``` is used, the train_test_split method will split the data equally every time the code runs.

```python
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X, y = load_airbnb()

std = StandardScaler()
X = std.fit_transform(X)

np.random.seed(10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

linear_regression_model_SDGRegr = SGDRegressor()

model = linear_regression_model_SDGRegr.fit(X_train, y_train)

y_pred = model.predict(x_test) # Price x Night Predictions
```
This task aims to get a baseline to compare other more advanced models and try to improve upon them.

In **Task 2**, using sklearn, the performance of the model is evaluated:

- Using the sklearn ```metrics```, the RMSE are R^2 are calculated:

    ```python
    from sklearn import metrics
    from sklearn.linear_model import SGDRegressor

    linear_regression_model_SDGRegr = SGDRegressor()
    model = linear_regression_model_SDGRegr.fit(X_train, y_train)

    y_pred = model.predict(x_test) 

    y_pred_train = model.predict(x_train) 

    print("Testing Metrics")

    print("RMSE: ")
    print(metrics.mean_squared_error(y_test, y_pred, squared=False))
    print("R^2: ")
    print(metrics.r2_score(y_test, y_pred))

    print(" ")

    print("Training Metrics")

    print("RMSE: ")
    print(metrics.mean_squared_error(y_train, y_pred_train, squared=False))
    print("R^2: ")
    print(metrics.r2_score(y_train, y_pred_train))
    ```

In **Task 3**, a function called ```custom_tune_regression_model_hyperparameters()``` is created. This function performs a grid search over a reasonable range of hyperparameter values to tune them.

SKLearn has methods like ```GridSearchCV``` to do this, but a similar method is created from scratch in this task.

The function takes in:

- ```models``` -> ```list``` : A list of models from sklearn in their abc.ABCMeta format

- ```X_train```, ```X_validation```, ```X_test``` -> ```numpy.ndarray``` : A set of numpy arrays containing the features of the model

- ```y_train```, ```y_validation```, ```y_test``` -> ```pandas.core.series.Series``` : A set of pandas series containing the targets/labels
        
- ```hyperparameters_dict``` -> ```list``` : A list of dictionaries containing a range of hyperparameters to be tried for each model

The function returns:

- ```best_regression_model``` -> ```sklearn.model``` : A model from sklearn

- ```best_hyperparameters_dict``` -> ```dict``` : A dictionary containing the best hyperparameter configuration

- ```best_metrics_dict``` -> ```dict``` : A dictionary containing the test metrics obtained using the best model  


The dictionary of performance metrics includes a variable called "validation_RMSE", for the RMSE on the validation set. As mentioned above, this value is used to select the best model. Then, the test metrics are used to score the model.

```python
def custom_tune_regression_model_hyperparameters(models, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
'''
    Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

    Parameters
    ----------
    models: list
        A list of models from sklearn in their abc.ABCMeta format

    X_train, X_validation, X_test: numpy.ndarray
        A set of numpy arrays containing the features of the model

    y_train, y_validation, y_test: pandas.core.series.Series
        A set of pandas series containing the targets/labels
    
    hyperparameters_dict: list
        A list of dictionaries containing a range of hyperparameters for each model

    Returns
    -------
    best_regression_model: sklearn.model
        A model from sklearn
    
    best_hyperparameters_dict: dict
        A dictionary containing the optimal hyperparameters configuration
    
    best_metrics_dict: dict 
        A dictionary containing the test metrics obtained using the best model         
'''

# Models input format : models = [SGDRegressor, linear_model.LinearRegression]

# Lists to store metrics, chosen Hyperparameters and the model for each iteration
validation_RMSE = []
validation_R2 = []
model_hyperparameters_val = [] 
model_val = []

# For each model, select the model class and the hyperparameters dictionary
for i in range(len(models)):
    model = models[i]
    hyperparameters_dict_ = hyperparameters_dict[i]

    # For each hyperparameter combination, create a model and store its metrics and hyperparameters
    for hyperparameters in itertools.product(*hyperparameters_dict_.values()):
        hyperparameters_ = dict(zip(hyperparameters_dict_.keys(),hyperparameters))
        regression_model = model(**hyperparameters_)
        model_ = regression_model.fit(X_train, y_train)
        y_pred = model_.predict(X_validation)
        validation_RMSE.append(metrics.mean_squared_error(y_validation, y_pred, squared=False))
        validation_R2.append(metrics.r2_score(y_validation, y_pred))
        model_hyperparameters_val.append(hyperparameters_)
        model_val.append(regression_model)

# Select the model with the best RMSE
index = np.argmin(validation_RMSE)
best_model = model_val[index]
best_hyperparameters_dict = model_hyperparameters_val[index]

# Train the best model
best_regression_model = best_model.fit(X_train, y_train)
y_pred_test= best_regression_model.predict(X_test)

# Obtain the metrics
test_RMSE = metrics.mean_squared_error(y_test, y_pred_test, squared=False)
test_R2 = metrics.r2_score(y_test, y_pred_test)

best_metrics_dict = {
    'RMSE' : test_RMSE,
    'R^2' : test_R2
}

return best_regression_model, best_hyperparameters_dict, best_metrics_dict
```

In **Task 4**, the hyperparameters of the model are tuned using methods from sklearn:

 - A function called ```tune_regression_model_hyperparameters()``` is created

The function takes in:

- ```model``` -> ```sklearn.model``` : An instance of the sklearn model

- ```X``` -> ```numpy.ndarray``` : A numpy array containing the features of the dataset

- ```y``` -> ```pandas.core.series.Series``` : A pandas series containing the test targets/labels of the dataset

- ```X_test``` -> ```numpy.ndarray``` : A numpy array containing the test features of the model

- ```y_test``` -> ```pandas.core.series.Series``` : A pandas series containing the test targets/labels
        
- ```hyperparameters_dict``` -> ```dict``` : A dictionary containing a range of hyperparameters to be tried for each model

The function returns:

- ```best_regression_model``` -> ```sklearn.model``` : A model from sklearn

- ```best_hyperparameters_dict``` -> ```dict``` : A dictionary containing the best hyperparameter configuration

- ```best_metrics_dict``` -> ```dict``` : A dictionary containing the test metrics obtained using the best model  

```python
def tune_regression_model_hyperparameters(model, X, y, X_test, y_test, hyperparameters_dict):
    '''
        Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

        Parameters
        ----------
        model: sklearn.model
            An instance of the sklearn model
        
        X: numpy.ndarray
            A numpy array containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels

        X_test: numpy.ndarray
            A numpy array containing the features of the model

        y_test: pandas.core.series.Series
            A pandas series containing the targets/labels
        
        hyperparameters_dict: dict
            A dictionary containing a range of hyperparameters 

        Returns
        -------
        best_regression_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model         
    '''
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}
    
    
    model = model
    hyperparameters = hyperparameters_dict
    grid_search = GridSearchCV(model, hyperparameters, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_hyperparameters_dict[model] = grid_search.best_params_
    best_metrics_dict[model] = grid_search.best_score_
    if best_regression_model is None or best_metrics_dict[model] > best_metrics_dict[best_regression_model]:
        best_regression_model = model
        best_hyperparameters = best_hyperparameters_dict[model]

    
    model = best_regression_model.fit(X,y)
    y_pred = model.predict(X_test)

    test_RMSE = (metrics.mean_squared_error(y_test, y_pred, squared=False))
    test_R2 = (metrics.r2_score(y_test, y_pred))
    best_metrics = {
    'RMSE' : test_RMSE,
    'R^2' : test_R2
    } 

    return best_regression_model, best_hyperparameters, best_metrics
```

In this case, the best model is chosen within the ```GridSearchCV``` method as it already splits the data into training and validation sets, so the model is chosen based on the validation metrics, and finally, the testing dataset is used to score the model's performance. The scoring criteria for selecting the best model within the ```GridSearchCV``` method is ```neg_mean_squared_error```.

In **Task 5**, the model is saved through the function ```save_model()```:

- Initially, this function creates a folder called ```models``` through Python's os library.

- Secondly, within the ```models``` folder, another folder called ```regression``` is created to save the regression models and their metrics in.

- Finally, within the ```regression``` folder, a last foler named ```folder_name``` is created to save the trained and tuned model in a file called ```model.joblib```, its hyperparameters in a file called ```hyperparameters.json```, and its performance metrics in a file called ```metrics.json```.

```python
def save_model(folder_name, best_model, best_hyperparameters, best_metrics):
    '''
        Creates a models folder, then within the models' folder creates a regression folder and finally makes a last folder_name folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics.
        
        Parameters
        ----------
        folder_name: str
            A string used to name the folder to be created
        
        best_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics: dict 
            A dictionary containing the test metrics obtained using the best model   

        Returns
        -------
        None
             
    '''

    # Create Models folder
    models_dir = 'airbnb-property-listings/models'
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if os.path.exists(models_path) == False:
        os.mkdir(models_path)

    # Create regression folder
    regression_dir = 'airbnb-property-listings/models/regression'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)
    if os.path.exists(regression_path) == False:
        os.mkdir(regression_path)

    # Create linear_regression folder
    folder_name_dir = os.path.join(regression_path,folder_name)
    current_dir = os.path.dirname(os.getcwd())
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if os.path.exists(folder_name_path) == False:
        os.mkdir(folder_name_path)

    # Save the model in a file called model.joblib
    joblib.dump(best_model, os.path.join(folder_name_path, 'model.joblib'))
   
    # Save the hyperparameters in a file called hyperparameters.json
    with open(os.path.join(folder_name_path, 'hyperparameters.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

    # Save the metrics in a file called metrics.json
    with open(os.path.join(folder_name_path, 'metrics.json'), 'w') as fp:
            json.dump(best_metrics, fp)


    return
```

In **Task 6**, the performance of the model is imporved by using different models provided by sklearn:

```python
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = [SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]
```

- A function called ```evaluate_all_models()``` uses the ```tune_regression_model_hyperparameters()``` function on each model to tune their hyperparameters before evaluating them. In addition, the best model, its hyperparameters, and its metrics are saved in a folder named after the model class 

```python
def evaluate_all_models(models,hyperparameters_dict):
    '''
        Imports and Standardizes the data, splits the dataset and finds the best-tuned model from the provided sklearn models and a range of its hyperparameters.       
        Finally, it saves the models, their metrics and their hyperparameters in their corresponding folders.
        
        Parameters 
        ----------
        models: list
            A list of models from sklearn 
        
        hyperparameters_dict: list
            A list of dictionaries containing a range of hyperparameters for each model

        Returns
        -------
        None
             
    '''

    # Import and standardize data
    X, y = import_and_standardize_data()

    # Split Data
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    # Tune models hyperparameters using GirdSearchCV
    for i in range(len(models)):

        best_regression_model, best_hyperparameters_dict, best_metrics_dict = tune_regression_model_hyperparameters(models[i], X, y, X_test, y_test, hyperparameters_dict[i])

        # Print Results
        print(best_regression_model, best_hyperparameters_dict, best_metrics_dict)

        # Save the models in their corresponding folders
        folder_name= str(models[i])[0:-2]
        save_model(folder_name, best_regression_model, best_hyperparameters_dict, best_metrics_dict)

    return
```

Finally, in **Task 7**, a function called ```find_best_model()``` evaluates which model is best, then returns the loaded model, a dictionary of its hyperparameters, and a dictionary of its performance metrics. This function iterates through the metrics.json files for each model and chooses the best model based on the highest ``` R^2```  score.


```python
def find_best_model(models):
    '''
        Using the metrics.json files produced in the evaluate_all_models(), this function iterates through the files to find the best metrics and output the best model, its hyperparameters and its metrics.

        Parameters 
        ----------
        models: list
            A list of models from sklearn 

        Returns
        -------
        best_regression_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model   
             
    '''

    # Find best metrics (best R^2 == highest score) within the libraries 
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}

    regression_dir = 'airbnb-property-listings/models/regression'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)
    
    for i in range(len(models)):
        model_str = str(models[i])[0:-2]
        model_dir = os.path.join(regression_path, model_str)
        model = load(os.path.join(model_dir, 'model.joblib'))
        hyperparameters_path = open(os.path.join(model_dir, 'hyperparameters.json'))
        hyperparameters = json.load(hyperparameters_path)
        metrics_path = open(os.path.join(model_dir, 'metrics.json'))
        metrics = json.load(metrics_path)

        if best_regression_model is None or metrics.get("R^2") > best_metrics_dict.get("R^2"):
            best_regression_model = model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict
```

This function inside a ```__name__ == "__main__"``` block, just after the ```evaluate_all_models()``` function.

```python
if __name__ == "__main__":

    evaluate_all_models(models, hyperparameters_dict)

    best_regression_model, best_hyperparameters_dict, best_metrics_dict = find_best_model(models)

    
    print("Best Regression Model:")
    print(best_regression_model)
    print("Hyperparameters:")
    print(best_hyperparameters_dict)
    print("Metrics:")
    print(best_metrics_dict)
```
