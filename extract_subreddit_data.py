import sqlite3
import pandas as pd

def get_subreddit_data():
    
    database_directory = input('Please input Reddit database directory:\nFor example: reddit_data.db\n')

    try:
        connection = sqlite3.connect(database_directory)

        print('\nQuerying database...\n')

        df = pd.read_sql_query(
            """
            SELECT 
                s.subreddit
                ,o.utc
                ,u.username 
            
            FROM Subreddit AS s
            
            JOIN Observation AS o
            ON (s.subreddit_id = o.subreddit_id)
            
            JOIN Username AS u
            ON (o.username_id = u.username_id)

            ORDER BY RANDOM()
            LIMIT 1000000
            """,
            connection
        )

        print('Sending extraction to csv file called "reddit_data.csv"\n')

        df.to_csv('./reddit_data.csv', index = False)

        print('\nDone!')
        
    except: 
        print('Reddit database does not exist in this directory')

get_subreddit_data()
