import numpy as np
import pandas as pd
import interface
import joblib
import warnings 
from reddit_recommender_engine import recommendation

# warnings.warn(msg, category=DeprecationWarning)


top_25_list = ['politics', 'The_Donald', 'nfl', 'leagueoflegends', 'worldnews',
       'funny', 'nba', 'pics', 'news', 'CFB', 'MMA', 'videos', 'SquaredCircle',
       'todayilearned', 'soccer', 'gaming', 'RocketLeagueExchange',
       'pcmasterrace', 'hockey', 'Overwatch', 'movies', 'AdviceAnimals',
       'DotA2', 'GlobalOffensive', 'gifs']


df = pd.read_csv('subreddit_pivot_table.csv')

labeled_dataset = pd.read_csv('labeled_dataset.csv')



subreddits = interface.get_subreddits(interface.get_number(top_25_list), df.drop('username', axis = 1))

subreddits_list = interface.get_subreddits_rating_list(subreddits, df.drop('username', axis = 1)) #Rey

model = joblib.load('subreddit_clustering_model.sav')

cluster = model.predict([subreddits_list])



subreddits_dictionary = (interface.get_rating_data_frame(subreddits, interface.get_user_name())) #Jose

recommendation(labeled_dataset, subreddits_dictionary, cluster[0])







