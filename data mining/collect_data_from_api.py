import pandas as pd
from urllib.request import urlopen
import json
import numpy as np
sys.path.append('../')
from configuration import URL_LIST

response_list = []
df_list = []

def get_data(link):
    with urlopen(link) as response:
        his = response.read()
        history = json.loads(his)
        response_list.append(history.get('data')['match'])
        if history.get('data')['next_page'] != False:
            get_data(history.get('data')['next_page'])

    for df in response_list:
    	df_list.append(pd.DataFrame(df))

    final_dataframe = pd.concat(df_list, axis=0)

   return final_dataframe


def expand_outcomes(link):
	df_history = get_wc_history_data(link)
	
	outcomes = df_history['outcomes'].apply(pd.Series)
	df_history_expanded = pd.concat([df_history, outcomes], axis=1)

	return df_history_expanded

# wc_history_df = expand_outcomes(wc_url)
# friendlies_df = expand_outcomes(friendlies_url)
# continents_df = expand_outcomes(continents_competitions_url)
# copa_america_df = expand_outcomes(copa_america_url)
# african_cup_df = expand_outcomes(african_cup_url)
# asian_cup_df = expand_outcomes(asian_cup_url)
# gold_cup_df = expand_outcomes(gold_cup_url)

def concat_all_dataframes(url_list):
	all_dataframes = []
	for url in url_list:
		df = expand_outcomes(url)
		all_dataframes.appned(df)

	all_history_dataframe = pd.concat(all_dataframes,ignore_index=True)

	all_history_dataframe.drop('outcomes',axis=1,inplace=True)

	return all_history_dataframe



def modify_dataframe(df_all_history):

	df_all_history['home_score'] = df_all_history['score'].str[0]
	df_all_history['away_score'] = df_all_history['score'].str[-1]

	df_all_history = df_all_history[df_all_history['home_score'] != '?']
	df_all_history.rename(columns={'away_name':'away_team','home_name':'home_team'},inplace=True)

	df_all_history = df_all_history[['competition_name','date','home_team','away_team','home_score','away_score','score','id']]

	# create a list of our conditions
	conditions = [
	    (df_all_history['home_score'] > df_all_history['away_score']),
	    (df_all_history['home_score'] < df_all_history['away_score'])
	    ]

	# create a list of the values we want to assign for each condition
	values = [1, 2]

	# create a new column and use np.select to assign values to it using our lists as arguments
	df_all_history['match_result'] = np.select(conditions, values,0)
	
	return df_all_history

def api_data_handler(url_list):

	try:
		df_all_history = concat_all_dataframes(url_list)

		modified_df = modify_dataframe(df_all_history)

		#TODO : change save path to AWS S3 bucket
		modified_df.to_csv('Model 2/Data/from_2017_matches_history.csv')
		return modified_df
	except Exception as e:
		print(e)






