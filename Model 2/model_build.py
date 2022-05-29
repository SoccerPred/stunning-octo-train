#Importing the necessary Libraries
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc , recall_score, precision_score , accuracy_score ,f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]


## KEY is file path in S3
def get_final_data_from_s3(KEY):

	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	obj = s3_client.get_object(Bucket= bucket , Key = KEY)
	df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')

	return df


def create_pipeline(categorical_cols,numerical_cols):
		#Use one hot encoder for categorical variables
	cat_pipeline = Pipeline([
	    ('select_cat', DataFrameSelector(categorical_cols)),
	    ('cat_encoder', OneHotEncoder(sparse=False))
	])

	#making dimensionality reduction to numerical columns and set the parameter to 'mle' Maximum Likelihood Estimation
	#to choose the best number of components
	num_pipeline = Pipeline([
	    ('select_numeric', DataFrameSelector(numerical_cols)),
	#     ('scaler', StandardScaler()),
	    ('pca',PCA(n_components='mle'))
	])

	#add the numerical and categorical variables together
	preprocess_pipeline = FeatureUnion(transformer_list=[
	    ('numeric_pipeline',num_pipeline),
	    ('cat_pipeline' , cat_pipeline)
	])

	return preprocess_pipeline

def process_final_data():
	df = get_final_data_from_s3('dump/final_df.csv')

	#rename column titles
	df.rename(columns={'home_team_Avg. age':'home_team. age'},inplace=True)
	df.rename(columns={'home_team':'home_team.name'},inplace=True)
	df.rename(columns={'away_team':'away_team.name'},inplace=True)

	#drop unnecessary columns as tournament, date and columns that had many null values as home_Poss, away_Poss,
	# I imputed the null values with the average value but after training the model it is better to drop them.
	df1 = df.drop(['goal_difference','home_score','away_score','date','home_Poss','away_Poss'],axis=1)

	#make 3 dataframes for home and away teams for later assigning data to each team name
	home_team_df = df1[['home_team.name', 'home_team.Rank', 'home_team. age',
	       'home_team.Total value', 'home_team.Points', 'home_Pl', 'home_Age',
	        'home_MP', 'home_Starts', 'home_Min', 'home_90s',
	       'home_Gls', 'home_Ast', 'home_G-PK', 'home_PK', 'home_PKatt',
	       'home_CrdY', 'home_CrdR', 'home_Gls.1', 'home_Ast.1', 'home_G+A',
	       'home_G-PK.1', 'home_G+A-PK']]

	away_team_df = df1[['away_team.name', 'away_team.Rank',
	       'away_team. age', 'away_team.Total value', 'away_team.Points',
	       'away_Pl', 'away_Age', 'away_MP', 'away_Starts',
	       'away_Min', 'away_90s', 'away_Gls', 'away_Ast', 'away_G-PK', 'away_PK',
	       'away_PKatt', 'away_CrdY', 'away_CrdR', 'away_Gls.1', 'away_Ast.1',
	       'away_G+A', 'away_G-PK.1', 'away_G+A-PK']]

	#set index as team name
	home_team_df = home_team_df.set_index(home_team_df['home_team.name']).drop('home_team.name',axis=1)
	away_team_df = away_team_df.set_index(away_team_df['away_team.name']).drop('away_team.name',axis=1)


	cols = home_team_df.columns
	cols2 = away_team_df.columns

	away_team_df.columns=cols
	df_2 = pd.concat([home_team_df, away_team_df], axis=0)

		
	away_team_df.columns=cols2
	home_team_df.columns=cols2

	df_3 = pd.concat([home_team_df, away_team_df], axis=0)


	#saving the dataframes to use later in the model file for deployment
	#TODO : Edit the save path
	df_2.to_csv('Data/df_home_all2.csv')
	df_3.to_csv('Data/df_away_all2.csv')

	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("Data/df_home_all2.csv", "dump/df_home_all2.csv")
	s3_client.Bucket(AWS_BUCKET_NAME).upload_file("Data/df_away_all2.csv", "dump/df_away_all2.csv")

	return df1, df2,df3 ,df



def building_the_model(df):
	df1 , _, _ ,_ = process_final_data()

	#separating the features and the target variables
	X = df1.drop('match_result',axis=1)
	y = df1['match_result']


	#building 2 lists for numerical and categorical columns for processing
	categorical_cols = [col for col in X.columns if (X[col].dtype == 'object')]

	numerical_cols = [col for col in X.columns if (X[col].dtype != 'object') ]

	preprocess_pipeline = create_pipeline(categorical_cols,numerical_cols)

	X = preprocess_pipeline.fit_transform(X)

	#splitting the data to 80% train and 20% test for features and target data
	X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

	#make cross validation with hyperparameters tuning for best performance using grid search.
	solvers = ['newton-cg', 'lbfgs', 'liblinear']
	penalty = ['l2','none']
	c_values = [300,100, 10, 1.0, 0.1, 0.01]
	# define grid search
	grid = dict(solver=solvers,penalty=penalty,C=c_values)
	model = LogisticRegression(max_iter=3000)

	#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)

	#fitting the model on the train data
	grid_search.fit(X_train_val , y_train_val)


	pickle_out = open("model2.pkl", "wb")
	pickle.dump(grid_search, pickle_out)
	pickle_out.close()

	pickle_out2 = open("pipeline2.pkl", "wb")
	pickle.dump(preprocess_pipeline, pickle_out2)
	pickle_out2.close()

	#adding to AWS S3
	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	pickle_byte_obj = pickle.dumps(grid_search)
	key1 = 'model2.pkl'
	s3_client.Object(bucket,key1).put(Body=pickle_byte_obj)

	pickle2_byte_obj = pickle.dumps(preprocess_pipeline)
	key2 = 'pipeline2.pkl.pkl'
	s3_client.Object(bucket,key2).put(Body=pickle2_byte_obj)

	return grid_search


def train_match_score_model():
	_, _, _ ,df = process_final_data()

	#drop unnecessary columns and columns with many null values as poss columns
	df_goals = df.drop(['goal_difference','match_result','date','home_Poss','away_Poss'],axis=1)


	#make a new columns with the final result from the home and away score columns
	df_goals['match_goals_result'] = df_goals['home_score'].apply(str)  + '-' + df_goals['away_score'].apply(str)

	#filter dataframe to exclude values with very low count values less than 10
	df_goals = df_goals[df_goals['match_goals_result'].isin(['1-0', '1-1', '0-0', '2-0', '0-1', '2-1', '1-2', '0-2', '3-0', '2-2',
	       '3-1', '4-0', '0-3', '1-3', '3-2', '2-3', '4-1', '5-0', '0-4', '1-4',
	       '6-0', '4-2', '5-1', '3-3', '2-4', '0-5', '6-1', '0-6', '4-3', '3-4',
	       '1-5'])]


	#drop the scores columns
	df_goals.drop(['home_score','away_score'],axis=1,inplace=True)

	#rename the columns to be consistent with previous model data
	df_goals.rename(columns={'home_team_Avg. age':'home_team. age'},inplace=True)
	df_goals.rename(columns={'home_team':'home_team.name'},inplace=True)
	df_goals.rename(columns={'away_team':'away_team.name'},inplace=True)

	#split data to features and target
	X_g = df_goals.drop('match_goals_result',axis=1)
	y_g = df_goals['match_goals_result']

	#building 2 lists for numerical and categorical columns for processing
	categorical_cols = [col for col in X_g.columns if (X_g[col].dtype == 'object')]

	numerical_cols = [col for col in X_g.columns if (X_g[col].dtype != 'object') ]
	
	preprocess_pipeline = create_pipeline(categorical_cols,numerical_cols) 
	#preprocess the features data using the preprocess_pipeline function
	X_g = preprocess_pipeline.fit_transform(X_g)

	#splitting the data to 80% train and 20% test for features and target data
	X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(X_g, y_g, test_size=0.20,random_state=42)

	#make cross validation with hyperparameters tuning for best performance using grid search.
	c_values = [300,10,0.1]
	# define grid search
	grid2 = dict(C=c_values)
	model2 = LogisticRegression(max_iter=3000 )

	grid_search2 = GridSearchCV(estimator=model2, param_grid=grid2, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)

	#fitting the model on the train data
	grid_search2.fit(X_train_val2 , y_train_val2)

	# pickling the model to use for deployment
	pickle_out = open("model_goals.pkl", "wb")
	pickle.dump(grid_search2, pickle_out)
	pickle_out.close()

	#adding to AWS S3
	s3_client = get_AWS_client('s3')
	bucket = s3_client.Bucket(AWS_BUCKET_NAME)

	pickle_byte_obj = pickle.dumps(grid_search2)
	key3 = 'model_goals.pkl'
	s3_client.Object(bucket,key3).put(Body=pickle_byte_obj)

	return grid_search2


def handle_model_building():
	building_the_model()
	train_match_score_model()




if __name__ == "__main__":
	handle_model_building()





