import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from functools import reduce
from sklearn.externals import joblib


def pipe(obj, *fns):
    '''
    This function will run every function through a pipeline process
    '''

    return reduce(lambda x, y: y(x), [obj] + list(fns))


def data_load_and_clean(data_file_directoy) -> pd.DataFrame:
    '''
    This function will take in a csv with columns [ubreddit, utc, username]

    It will then filter out outliers

    Lastly, it will return a dataframe filtered for non-outlier subreddits
    '''

    print('Loading dataset...\n')
    df = pd.read_csv(data_file_directoy)

    print('Cleaning dataset...\n')
    #First filter logic
    count = pd.DataFrame(df['subreddit'].value_counts())
    above_1000 = count[(count['subreddit']> 1000) & (count['subreddit']< 400000)]
    clean_list = list(above_1000.index)

    #Remove largest subreddit
    clean_df = df[df['subreddit'] != 'AskReddit']

    #Filter table for subreddits that were selected in the above range
    return clean_df[clean_df['subreddit'].str.contains('|'.join(clean_list))]


def build_datastructure(clean_dataframe : pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes a dataframe and calculates ratings by user for their respective subreddit

    It then removes users who only have 1 subreddit interaction (making their interest rating 1) which would not be useful for our recommendations

    It finally returns a dataframe with users as 1 column and each subreddit is a column with the user/subreddit pair having their rating
    '''
    
    print('Building 1st data structure...\n')
    agg_df = clean_dataframe.pivot_table(index = ['username','subreddit'], aggfunc = 'count').sort_values(by='username', ascending=False)

    user_df = pd.DataFrame(agg_df.groupby('username')['utc'].sum())

    #Merge data gathered with original dataframe
    final_df = pd.merge(agg_df.reset_index(),user_df,on='username',how='left')

    #Get a rating score
    final_df['rating'] = final_df['utc_x'] / final_df['utc_y']

    #Get final dataframe (necessary columns)
    final_df = final_df[['username', 'subreddit', 'rating']]

    #Remove users that have one subreddit 
    final_df = final_df[final_df['rating'] < 1]

    #Get dataframe structure ready for the model
    return final_df.pivot_table(index = 'username', columns = 'subreddit', values = 'rating', aggfunc = 'max').fillna(0).reset_index(), final_df


def fit_model(clean_pivot : pd.DataFrame):
    '''
    This function fits the clean dataframe to train the model

    It returns a trained model

    It saves the model in the current directory to be used when predicting the user's inputs
    '''

    print('\nTraining model...\n')
    kmeans = KMeans(n_clusters = 25)
    subreddit_clustering_model = kmeans.fit(clean_pivot.drop('username', axis = 1))

    print('Sending saved model to file called "subreddit_clustering_model.sav"...\n')
    file_name = 'subreddit_clustering_model.sav'

    joblib.dump(subreddit_clustering_model, file_name)

    return subreddit_clustering_model


def join_labels(trained_model, train_df_orig, train_df):
    '''
    This function takes a trained model

    Predicts the labels for our data 

    Merges the labels to the original dataframe
    '''

    #Attaching the labels back to original dataframe
    train_df_orig['labels'] = trained_model.fit_predict(train_df_orig.drop('username', axis = 1))

    train_merged = train_df.merge(train_df_orig[['username','labels']], on = 'username', how = 'left')

    print('Sending labeled data to CSV file called "labeled_dataset"...')
    train_merged.to_csv('labeled_dataset.csv', index = False)


def run_subreddit_clustering_program():
    '''
    '''

    modeling_datastructure, master_dataframe = pipe(data_load_and_clean('reddit_data.csv'),
        build_datastructure
    )
    
    print('\nSending subreddit table to a csv called "subreddit_table"')
    modeling_datastructure.to_csv('subreddit_pivot_table.csv', index = False)
    
    model = fit_model(modeling_datastructure)

    return join_labels(model, modeling_datastructure, master_dataframe)


print('\nRunning clustering program...\n')
run_subreddit_clustering_program()