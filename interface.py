import numpy as np
import pandas as pd

def get_user_name():
    '''
        Getting the username that wil be use it in the recommendation system
    '''
    x = input("Please, enter your username:\n")
    return x

def get_number(top_25_list):
    '''
        Getting the number of subredits that the user likes
    '''
    print("Hello user, welcome to our great recommender program!\n\nHere are some popular subreddits if you are having trouble selecting:\n\n" , top_25_list)
    while True:
        try:
            x = int(input("How many subreddits do you want to give our recommender system? Please enter a number:\n"))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    return x

def get_subreddits(x, df):
    '''
        Processing the subreddits

    '''
    d = {}
    n = x
    df.columns = map(str.lower, df.columns)
    columns = [e.lower() for e in df.columns]
    print("Please type your subreddits considering the priority. Start with what you like more.")
    while n>0:
        subreddit = input("\nEnter a subreddit:\n").lower()
        if subreddit in columns:
            keys = subreddit
            values = n/x - 0.05
            d[keys] = values
            n -= 1
        else:
            print("Your subreddit it's not in our data, please enter the subreddit again.")
    return d

def get_subreddits_rating_list(d, df): # to Ray
    '''
    Returning a list of ratings by subreddits
    (Kipping the consistency between the position of the elements in the list and trainning data columns)

    This is the list that will be use it to predict
    '''     
    new_df = pd.DataFrame()
    new_df = df.append(d, ignore_index=True).fillna(0)
    return list(new_df.iloc[-1].values)

def get_rating_data_frame(d, user_name): # to Jose
    '''
    Returning a dataframe - ratings by subreddits
    
    This is the dataframe that will be used in the recommender system
    ''' 
    new_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
    new_df['username'] =  user_name
    new_df.rename(columns={"index": "subreddit", 0: "rating"}, inplace = True)
    return new_df










