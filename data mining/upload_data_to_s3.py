import boto3

from collect_data_from_api import api_data_handler
from collect_data_stats import stats_data_handler
from collect_rank_data import df_handler

sys.path.append('../')
from configuration import *
from utils_aws import *


def collecting_data():

	hist_df = api_data_handler(URL_LIST)
	stats_df = stats_data_handler(2014_url,2010_url,2018_url,2022_friendlies,2021_friendlies,2020_friendlies)
	rank_df = df_handler(PAGE_URL)

	s3_client = get_AWS_client('s3')

	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("/Model 2/Data/from_2017_matches_history.csv", "dump/hist.csv")
	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("/Model 2/Data/team_avg_statistics_including_friendly.csv", "dump/stats.csv")
	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("/Model 2/Data/rank.csv", "dump/rank.csv")

