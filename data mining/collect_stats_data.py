import pandas as pd
sys.path.append('../')
from configuration import 2014_url , 2010_url,2018_url,2022_friendlies,2021_friendlies,2020_friendlies , DF_COLUMNS

def process_df(df):
	df.columns = df.columns.droplevel(0)
	df.rename(columns={'# Pl' : 'Pl'},inplace=True)
	df['Squad'] = df['Squad'].str[3:]
	df.replace(' England','England',inplace=True)

def generate_final_df(2014_url,2010_url,2018_url,2022_friendlies,2021_friendlies,2020_friendlies):

	df2014 = pd.read_html(2014_url)
	df2014_teams = process_df(df2014)

	df2010 = pd.read_html(2010_url)
	df2010_teams = process_df(df2010)

	df2018 = pd.read_html(2018_url)
	df2018_teams = process_df(df2018)
	df2018_teams = df2018_teams.iloc[:,:-9]

	df2022_friendlies = pd.read_html(2022_friendlies_url)
	df2022_friendlies = process_df(df2022_friendlies)

	df2021_friendlies = pd.read_html(2021_friendlies_url)
	df2021_friendlies = process_df(df2021_friendlies)

	df2020_friendlies = pd.read_html(2020_friendlies_url)
	df2020_friendlies = process_df(df2020_friendlies)

	df_wc_concat = pd.concat([df2010_teams,df2014_teams, df2018_teams])

    df_friendly_stats = pd.concat([df2020_f,df2021_f, df2022_f])

    return df_wc_concat,df_friendly_stats

def clean_dfs(df_wc_concat,df_friendly_stats):

	df_wc_concat.columns = DF_COLUMNS

    df_wc = df_wc_concat.groupby('Squad').mean().reset_index()

    df_friendly_stats.columns = DF_COLUMNS

    df_friendly = df_friendly_stats.groupby('Squad').mean().reset_index()


    df_wc = df_wc.replace('IR Iran','Iran')

	df_wc['Poss'].fillna(df_wc['Poss'].mean(),inplace=True)
	df_wc.loc['mean'] = df_wc.mean()
	df_wc.rename({df_wc.index[-1]:50}, inplace=True)

	# remove leading space
	df_friendly['Squad'] = df_friendly['Squad'].str.lstrip()

	df_all = pd.concat([df_wc,df_friendly])
	df_all =  df_all.groupby('Squad').mean().reset_index()

	return df_all


def stats_data_handler(2014_url,2010_url,2018_url,2022_friendlies,2021_friendlies,2020_friendlies):

	try:
		df_wc,df_friendly_stats = generate_final_df(2014_url,2010_url,2018_url,2022_friendlies,2021_friendlies,2020_friendlies)

		df_all = clean_dfs(df_wc,df_friendly_stats)
		#TODO : replace save path to AWS S3 bucket
		df_all.to_csv('Model 2/Data/team_avg_statistics_including_friendly.csv')

		return df_all

	except Exception as e:
		print(e)




