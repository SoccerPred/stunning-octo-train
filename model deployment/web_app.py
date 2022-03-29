import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]


# loading in the model to predict on the data
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)
classes = model.classes_

pickle_in3 = open('pipeline.pkl', 'rb')
pipeline = pickle.load(pickle_in3)


team1_list = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Colombia', 'Costa Rica',
       'Croatia', 'Denmark', 'Egypt', 'England', 'France', 'Germany',
       'Iceland', 'Iran', 'Japan', 'Mexico', 'Morocco', 'Nigeria', 'Panama',
       'Peru', 'Poland', 'Portugal', 'Russia', 'Saudi Arabia', 'Senegal',
       'Serbia', 'Spain', 'Sweden', 'Switzerland', 'Tunisia', 'Uruguay']

team2_list = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Colombia', 'Costa Rica',
       'Croatia', 'Denmark', 'Egypt', 'England', 'France',
       'Iceland', 'Iran', 'Japan', 'Mexico', 'Morocco', 'Nigeria', 'Panama',
       'Peru', 'Poland', 'Portugal', 'Russia', 'Saudi Arabia', 'Senegal',
       'Serbia', 'Spain', 'Sweden', 'Switzerland', 'Tunisia', 'Uruguay']

df_2 = pd.read_csv('df_home_all.csv',index_col=0)
df_3 = pd.read_csv('df_away_all.csv',index_col=0)

                
def welcome():
	return 'welcome all'


def prediction(processed_values):

        
	prediction = model.predict(processed_values)
	
	return prediction


# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">World Cup match Prediction App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	team_1 = st.selectbox('Team 1', np.array(team1_list))
	team_2 = st.selectbox('Team 2', np.array(team2_list))

	
#	results_df = predict_match_result(team_1 , team_2)

        # the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	results_df = pd.DataFrame()
	
	if st.button("Predict"):
            if (team_1 == team_2):
                st.text('Please add different teams')
            else:
                results_df = predict_match_result(team_1 , team_2)
            
                st.dataframe(results_df)	    

	
	
def assign_values_to_team1(team):
    
    if team in df_2.index :
        team1_data =  df_2.loc[team].reset_index()
        team1_data = team1_data.groupby('index').mean().reset_index().rename(columns={'index':'home_team.name'}).iloc[0]
        return team1_data


def assign_values_to_team2(team):
    
    if team in df_3.index :
        team2_data =  df_3.loc[team].reset_index()
        team2_data = team2_data.groupby('index').mean().reset_index().rename(columns={'index':'away_team.name'}).iloc[0]
        return team2_data

    
def map_inputs_to_data(team1,team2):

    team_1z = assign_values_to_team1(team1)
               
    team_2z = assign_values_to_team2(team2)

    input_data = pd.concat([team_1z,team_2z])
    return input_data


def predict_match_result(team1 ,team2):


    input_d = map_inputs_to_data(team1 , team2)
    input_processed = pipeline.transform(pd.DataFrame(input_d).T)
    preds_test = model.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes,data=np.round(preds_test,3))
    results_df.rename(columns={0:'Draw Probability',1:'{} wins Probability'.format(team1),2:'{} wins Probability'.format(team2)},inplace=True)
    return results_df
    
#    else:
#        return '' 

	
if __name__=='__main__':
	main()

