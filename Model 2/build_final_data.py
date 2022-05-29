import pandas as pd
import numpy as np
import boto3
import io

sys.path.append('../')
from configuration import *
from utils_aws import *

## KEY is file path in S3
def get_data_from_s3(KEY):

	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	obj = s3_client.get_object(Bucket= bucket , Key = KEY)
	df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')

	return df

def get_and_modify_data():

	hist_df = get_data_from_s3("dump/hist.csv")
	stats_df = get_data_from_s3("dump/stats.csv")
	rank_df = get_data_from_s3("dump/rank.csv")

	hist_df = hist_df.drop('Unnamed: 0',axis=1)
	stats_df = stats_df.drop('Unnamed: 0',axis=1)
	rank_df = rank_df.drop('Unnamed: 0',axis=1)

	return hist_df , stats_df , rank_df


def process_stats_data(team_stats):
	
	#replacing different team names 
	team_stats.replace({'United States':'USA'},inplace=True)
	team_stats.replace({'Korea Republic':'South Korea'},inplace=True)

	return team_stats


def process_hist_data(df_api):

	# create a list of our conditions
	conditions = [
	    (df_api['home_score'] > df_api['away_score']),
	    (df_api['home_score'] < df_api['away_score'])
	    ]

	# create a list of the values we want to assign for each condition
	values = [df_api['home_team'], df_api['away_team']]

	# create a new column and use np.select to assign values to it using our lists as arguments
	df_api['winning_team'] = np.select(conditions, values,'Draw')


	df_api['goal_difference']=np.abs(df_api['home_score'] - df_api['away_score'])

	df_api.drop('score',axis=1,inplace=True)
	sorter = ['date', 'home_team', 'away_team', 'competition_name', 'home_score',
	       'away_score', 'winning_team', 'goal_difference', 'match_result']

	df_api = df_api[sorter]


	return df_api

def add_rank_data_to_history_data(fifa_rank,df_api):

	df_all = process_hist_data(df_api)

	#replace the only different team name
	fifa_rank.replace({'U. A. E.':'United Arab Emirates'},inplace=True)
	#adding the statistics data to both teams and make home and away datasets to merge later
	df_rank_home = fifa_rank.rename(columns={'Nation':'home_team','Avg. age':'home_team_Avg. age','Total value':'home_team.Total value','Points':'home_team.Points','Rank':'home_team.Rank'}
	                    )
	df_rank_away = fifa_rank.rename(columns={'Nation':'away_team','Avg. age':'away_team. age','Total value':'away_team.Total value','Points':'away_team.Points','Rank':'away_team.Rank'}
	                    )

	new_df = pd.merge(df_all,df_rank_home, on='home_team')
	new_df = pd.merge(new_df,df_rank_away, on='away_team')

	#Drop unnecessary columns
	new_df.drop(['Confederation_x','Confederation_y','Squad size_y','Squad size_x'],axis=1,inplace=True)

	return new_df

def add_all_data(team_stats,df_api,fifa_rank):
	
	df = add_rank_data_to_history_data(fifa_rank,df_api)
	team_stats = process_stats_data(team_stats)

	#adding prefix for away team 
	df_team_away = team_stats.add_prefix('away_')
	df_team_away.rename(columns={'away_Squad':'away_team'},inplace=True)

	#adding prefix for home team 
	df_team_home = team_stats.add_prefix('home_')
	df_team_home.rename(columns={'home_Squad':'home_team'},inplace=True)

	df_all = pd.merge(df,df_team_home, on='home_team')
	df_all = pd.merge(df_all , df_team_away , on='away_team')

	#Drop the Winning team column and keep the result column which indicates if home time Wins, Lose or Draw with (W,L,D)
	df_all.drop('winning_team',axis=1,inplace=True)
	df_all['date'] = pd.to_datetime(df_all['date'])


	sorter = ['date','home_team','home_score','home_team.Rank','home_team_Avg. age','home_team.Total value', 'home_team.Points',
         'home_Pl', 'home_Age', 'home_Poss', 'home_MP', 'home_Starts','home_Min', 'home_90s', 'home_Gls', 'home_Ast', 'home_G-PK', 'home_PK',
       'home_PKatt', 'home_CrdY', 'home_CrdR', 'home_Gls.1', 'home_Ast.1','home_G+A', 'home_G-PK.1', 'home_G+A-PK',
         'away_team','away_score','away_team.Rank','away_team. age', 'away_team.Total value', 'away_team.Points', 'away_Pl', 'away_Age',
       'away_Poss', 'away_MP', 'away_Starts', 'away_Min', 'away_90s','away_Gls', 'away_Ast', 'away_G-PK', 'away_PK', 'away_PKatt',
       'away_CrdY', 'away_CrdR', 'away_Gls.1', 'away_Ast.1', 'away_G+A','away_G-PK.1', 'away_G+A-PK','goal_difference', 'match_result'
         ]

	df_all = df_all[sorter]

	#save the data to a csv file
	df_all.to_csv('Data/final_df.csv')



	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("Data/final_df.csv", "dump/final_df.csv")

	return df_all


def handle_data_and_upload_s3():

	hist_df , stats_df , rank_df = get_and_modify_data()
	all_data = add_all_data(stats_df,hist_df,fifa_rank)

	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("Data/final_df.csv", "dump/final_df.csv")


if __name__ == "__main__":
	handle_data_and_upload_s3()



