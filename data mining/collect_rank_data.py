import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
sys.path.append('../')
from configuration import HEADERS , PAGE_URL



total_values = []
ranks_list = []
sqaud_size = []
avg_age = []
points = []
teams_list = []
confederations = []

def scrape_rank_data(page):
	for page_number in range(1,10):
    page_link = page.format(page_number)
    pageTree = requests.get(page_link, headers=HEADERS)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
    
    ranks = pageSoup.find_all("td", {"class": "zentriert cp"})
    teams = pageSoup.find_all("td", {"class": "hauptlink"})
    squad = pageSoup.find_all("td", {"class": "zentriert"})
    t_value = pageSoup.find_all("td", {"class": "rechts"})


    for i in range(0,len(squad),5):
        sqaud_size.append(squad[i+1].text)   
        avg_age.append(squad[i+2].text)
        confederations.append(squad[i+3].text)
        points.append(squad[i+4].text) 

    for i in t_value:
        total_values.append(i.get_text())


    for i in ranks:
        ranks_list.append(i.get_text())

    for i in range(0,len(teams),2):
        teams_list.append(teams[i].text[1:])


    rank_df = pd.DataFrame({'Rank':ranks_list,
                       'Nation':teams_list,
                       'Squad size':sqaud_size,
                       'Avg. age':avg_age,
                       'Total value':total_values,
                       'Confederation':confederations,
                       'Points':points})
    return rank_df


def modify_rank_df(rank_df):
	rank_df = rank_df.replace('-','00')
	rank_df['Total value'] = [int(1000000000*float(x.replace('bn', ''))) if 'bn' in x else int(1000000*float(x.replace('m', ''))) if 'm' in x else int(1000*float(x.replace('Th.',''))) if 'Th.' in x else float(x) for x in rank_df['Total value'].str[1:]]

	rank_df['Rank'] = pd.to_numeric(rank_df['Rank'].str[:-1],errors='coerce').astype(np.int64)
	rank_df['Squad size'] = pd.to_numeric(rank_df['Squad size'],errors='coerce').astype(np.int64)
	rank_df['Avg. age'] = pd.to_numeric(rank_df['Avg. age'],errors='coerce')
	rank_df['Points'] = pd.to_numeric(rank_df['Points'],errors='coerce').astype(np.int64)

	return rank_df

def df_handler(page):
	try:
		rank_df = scrape_rank_data(page)
		modified_df = modify_rank_df(rank_df)
		modified_df.to_csv('/Model 2/Data/rank.csv')

		return modified_df

	except Exception as e:
		print(e)

