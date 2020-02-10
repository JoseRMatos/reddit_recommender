import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def recommendation(cluster_df,client_df, client_cluster, num_rec = 10):
    
    # filters the dataframe based on the client cluster
    client_group = cluster_df[cluster_df['labels'] == client_cluster] 

    # Get list of usernames with that have try the subreddit selected by client
    username_list = list(client_group[client_group['subreddit']
                                      .isin(list(client_df['subreddit']))]['username'].unique()) 
    
    # Filters module by only the users that have same subreddit
    new = client_group[client_group['username'].isin(username_list)]
    new = new.drop(columns = ['labels'])
    
    # Adding customers selections to the frame
    selection = pd.concat([new, client_df], sort=True)
    dft = pd.pivot_table(selection, values ='rating', index ='username', 
                   columns ='subreddit', aggfunc = np.sum, fill_value=0)
    
    # Distance between users
    sqform = pd.DataFrame(squareform(pdist(dft)))
    sqform.columns = sqform.index = dft.index
    similarities = 1 / (1 +sqform)
    
    # Getting the weight (correlation) of other users and dropping the client
    client_weights = (similarities[[client_df['username'][0]]]
                      .transpose()
                      .drop(columns = client_df['username'][0])
                     )
    
    # Matrix of client values of their subreddits - clients as columns and s
    new = pd.pivot_table(new, values ='rating', index ='subreddit', 
                   columns ='username', aggfunc = np.sum, fill_value=0)
    # Creating new dataframe based on the distance of previous users and the weigth of previous users
    recommendations = pd.DataFrame()
    for name in client_weights:
        recommendations[name] = new[name] * client_weights[name].values
    
    # DataFrame by total and adding percentage value
    recommendations['Total'] = recommendations.sum(axis = 1)
    df = recommendations.sort_values('Total', ascending = False)
    
    df['Percentage'] = (df['Total']/df['Total'].max(axis = 0)*100).round(2).astype(str)  + ' %'
    
    #Removing client subreddit to offer new ones
    df = df[~df.index.isin(list(client_df['subreddit']))]
    
    print('Here are our top 10 recommendations for you:')
    print(pd.DataFrame(df.index[:num_rec]))