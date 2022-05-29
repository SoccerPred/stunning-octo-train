


## API international history data Urls ##
WC_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=362'
FRIENDLIES_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=371&from=2010-01-01'
CONTINENTS_COMPETITION_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=387&from=2010-01-01'
COPA_AMERICA_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=271&from=2010-01-01'
AFRICA_CUP_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=227&from=2010-01-01'
ASIAN_CUP_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=240&from=2010-01-01'
GOLD_CUP_URL = 'http://livescore-api.com/api-client/scores/history.json?key=sAyccMI0kNtc6o2t&secret=qDty37iEy2ENubtXL2aYOTY9GslMQogh&competition_id=266&from=2010-01-01'

URL_LIST = [WC_URL,FRIENDLIES_URL,CONTINENTS_COMPETITION_URL,COPA_AMERICA_URL,AFRICA_CUP_URL,ASIAN_CUP_URL,GOLD_CUP_URL]


## Stats data Urls ##
2014_url = 'https://fbref.com/en/comps/1/1709/stats/2014-FIFA-World-Cup-Stats'
2010_url = 'https://fbref.com/en/comps/1/19/stats/2010-FIFA-World-Cup-Stats'
2018_url = 'https://fbref.com/en/comps/1/stats/FIFA-World-Cup-Stats'

2022_friendlies_url = 'https://fbref.com/en/comps/218/stats/Friendlies-M-Stats'
2021_friendlies_url = 'https://fbref.com/en/comps/218/10979/stats/2021-Friendlies-M-Stats'
2020_friendlies_url = 'https://fbref.com/en/comps/218/3697/stats/2020-Friendlies-M-Stats'


DF_COLUMNS = ['Squad', 'Pl', 'Age', 'Poss', 'MP', 'Starts', 'Min', '90s', 'Gls',
       'Ast', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1', 'Ast.1', 'G+A',
       'G-PK.1', 'G+A-PK']


## RANK DATA ##

HEADERS = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
PAGE_URL = "https://www.transfermarkt.com/wettbewerbe/fifa?ajax=yw1&page={}"


############################################
# AWS
############################################
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION_NAME = ''
AWS_BUCKET_NAME = ''
