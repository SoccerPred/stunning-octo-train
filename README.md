# stunning-octo-train

Soccer World cup prediction app

## Data Sources:
• **Teams General statistics** from https://fbref.com, which includes teams statistics for :
  1. World cup 2018 (https://fbref.com/en/comps/1/stats/FIFA-World-Cup-Stats)
  2. World cup 2014 (https://fbref.com/en/comps/1/1709/stats/2014-FIFA-World-Cup-Stats)
  3. world cup 2010 (https://fbref.com/en/comps/1/19/stats/2010-FIFA-World-Cup-Stats)
  4. Friendly macthes for years 2020-2021-2022 
    (https://fbref.com/en/comps/218/3697/stats/2020-Friendlies-M-Stats)
    (https://fbref.com/en/comps/218/10979/stats/2021-Friendlies-M-Stats)
    (https://fbref.com/en/comps/218/stats/Friendlies-M-Stats)

#### The main features are :
  * PI : number of players
  * MP : matches played
  * 90s : minutes played divided by 90
  * GLs : goals
  * Ast : assistance
  * G-PK: non - penalty goals
  * PK :penalty kicks made
  * PKatt : penalty kicks attempted
  * CrdY : yellow cards
  * CrdR : red cards
  * G+A : goals + assists
  * G-PK.1 :goals minus penalty kicks
  * G+A-PK : goals + assists minus penalty kicks

For dupliaces team names i used the average statistics of different datasets.

• **Matches Results from 2010 to 2018 from kaggle**
  (https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
  and i filtered the data to contain only data from 2010
  
• **Match results from 2017 to 2022 from API** (data in the API are only from 2017): 
  (http://livescore-api.com/)
  
  I collected the history data for world cup 2018, friendly matches, all continents competitions as (UEFA,Copa America, Asian Cup, etc...)
  
 • **Fifa Rank data** (from https://www.transfermarkt.com/statistik/weltrangliste)
  The data contains:
  1. Team rank
  2. Squad size
  3. Total value (in Euros)
  4. Average Age
  5. Points

The matches results data from 2010 to 2018 was merged with the new data from API to get all matches data from 2010 to 2022, and added the rank data and average teams statistics to each team in one dataframe



