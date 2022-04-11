"""The main file to predict soccer world cup predictions

The code consists of 3 main parts:
1. Assigning each team data from the csv files to eact team name, and build a dataframe similar to the one the model is trained on.

2. Loading the model and the Preprocess pipeline to process the data including (one hot encoding for categorical variables and Making
PCA for the numerical values for dimensionality reduction

3.Making a web app front end frame work using Streamlit library to deploy the model and put it into production
"""

#Importing the libraries
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


#loading in the model and the pipeline files to predict on the data for the first model
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)
classes = model.classes_

pickle_in3 = open('pipeline.pkl', 'rb')
pipeline = pickle.load(pickle_in3)

#loading in the model and the pipeline files to predict on the data for the second model
pickle_in4 = open('model2.pkl', 'rb')
model2 = pickle.load(pickle_in4)
classes2 = model2.classes_

pickle_in5 = open('pipeline2.pkl', 'rb')
pipeline2 = pickle.load(pickle_in5)

pickle_in6 = open('model_goals.pkl', 'rb')
model3 = pickle.load(pickle_in6)
classes3 = model3.classes_


#create choose list for first model including the teams names from the trained data
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

#create choose list for second model including the teams names from the trained data
team1_list2 = ['Algeria', 'Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon',
       'Canada','Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Denmark', 'Ecuador',
       'Egypt', 'England', 'France', 'Germany', 'Ghana', 'Greece', 'Honduras',
       'Iceland', 'Iran', 'Italy', 'Japan','Mexico', 'Morocco', 'Netherlands',
       'New Zealand', 'Nigeria', 'Panama', 'Paraguay', 'Peru', 'Poland',
       'Portugal', 'Qatar','Russia', 'Saudi Arabia', 'Scotland','Senegal', 'Serbia', 'Slovakia',
       'Slovenia', 'South Africa', 'South Korea','Spain', 'Sweden', 'Switzerland', 'Tunisia',
       'Uruguay', 'USA', 'Ukraine', 'United Arab Emirates','Wales']

team2_list2 = team1_list2.copy()


#read the meta data for both home and away teams to assign the data
#based on the choosen team for the first model
df_2 = pd.read_csv('df_home_all.csv',index_col=0)
df_3 = pd.read_csv('df_away_all.csv',index_col=0)

#read the meta data for both home and away teams to assign the data
#based on the choosen team for the second model
df_home = pd.read_csv('df_home_all2.csv',index_col=0)
df_away = pd.read_csv('df_away_all2.csv',index_col=0)
                
def welcome():
	return 'welcome all'


# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">World Cup match Prediction App </h1>
	<h3 style ="color:black;text-align:center;">First model</h3>
	</div>
	"""
	
	# this line allows us to display a drop list to choose team 1 and team 2 
	st.markdown(html_temp, unsafe_allow_html = True)
	team_1 = st.selectbox('Team 1', np.array(team1_list))
	team_2 = st.selectbox('Team 2', np.array(team2_list))

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


	#add a header of the second model
	html_temp2 = """
        <div style ="background-color:yellow;padding:10px">
	<h3 style ="color:black;text-align:center;">Second model</h3>
	</div>
	"""
	
	# this line allows us to display a drop list to choose team 1 and team 2 
	st.markdown(html_temp2, unsafe_allow_html = True)
	team_3 = st.selectbox('Team 1', np.array(team1_list2))
	team_4 = st.selectbox('Team 2', np.array(team2_list2))


        # the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	results_df2 = pd.DataFrame()
	
	if st.button("Predict "):
            if (team_3 == team_4):
                st.text('Please add different teams')
            else:
                
                results_df2 = predict_match_result2(team_3 , team_4)
                
                st.dataframe(results_df2)
                #this step to preduict the match final result and display the highest results propabilities
                draw_df , home_w_df , away_w_df = predict_match_result_goals(team_3 , team_4)

                
                st.markdown('Match Result prediction')

                
                #add three dataframes of the match results in case of Draw, Win , Lose
                col1, col2 ,col3 = st.columns(3)
                col1.markdown("Draw Results")
                col1.dataframe((pd.DataFrame(draw_df.loc[0].nlargest(3)).T)*(1/(draw_df.loc[0].nlargest(3).values.sum())))
                col2.markdown("Team 1 win Results")
                col2.dataframe((pd.DataFrame(home_w_df.loc[0].nlargest(3)).T)*(1/(home_w_df.loc[0].nlargest(3).values.sum())))
                col3.markdown("Team 2 win Results")
                col3.dataframe((pd.DataFrame(away_w_df.loc[0].nlargest(3)).T)*(1/(away_w_df.loc[0].nlargest(3).values.sum())))

	
#functions for model 1
#Assign values from the dataframe to the team name and retuen a dataframe with all team1 data
def assign_values_to_team1(team):
    
    if team in df_2.index :
        team1_data =  df_2.loc[team].reset_index()
        team1_data = team1_data.groupby('index').mean().reset_index().rename(columns={'index':'home_team.name'}).iloc[0]
        return team1_data

#Assign values from the dataframe to the team name and retuen a dataframe with all team2 data
def assign_values_to_team2(team):
    
    if team in df_3.index :
        team2_data =  df_3.loc[team].reset_index()
        team2_data = team2_data.groupby('index').mean().reset_index().rename(columns={'index':'away_team.name'}).iloc[0]
        return team2_data


#run the assign values functions and concat the resultiung 2 dataframes into one dataframe for the model input
def map_inputs_to_data(team1,team2):

    team_1z = assign_values_to_team1(team1)
               
    team_2z = assign_values_to_team2(team2)

    input_data = pd.concat([team_1z,team_2z])
    return input_data


#get the input data and preprocess the data using the loaded data processing Pipeline,
#and predict the match result probabilities using predict_proba function, and return a dataframe with the probabilites.
def predict_match_result(team1 ,team2):


    input_d = map_inputs_to_data(team1 , team2)
    input_processed = pipeline.transform(pd.DataFrame(input_d).T)
    preds_test = model.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes,data=np.round(preds_test,3))
    results_df.rename(columns={0:'Draw Probability',1:'{} wins Probability'.format(team1),2:'{} wins Probability'.format(team2)},inplace=True)
    return results_df

#functions for model2 data assignment
#Assign values from the dataframe to the team name and retuen a dataframe with all team1 data
def assign_values_to_team3(team):
    
    if team in df_home.index :
        team1_data =  df_home.loc[team].reset_index()
        team1_data = team1_data.groupby('index').mean().reset_index().rename(columns={'index':'home_team.name'}).iloc[0]
        return team1_data

#Assign values from the dataframe to the team name and retuen a dataframe with all team2 data
def assign_values_to_team4(team):
    
    if team in df_away.index :
        team2_data =  df_away.loc[team].reset_index()
        team2_data = team2_data.groupby('index').mean().reset_index().rename(columns={'index':'away_team.name'}).iloc[0]
        return team2_data

#run the assign values functions and concat the resultiung 2 dataframes into one dataframe for the model input
def map_inputs_to_data2(team1,team2):

    team_3z = assign_values_to_team3(team1)
               
    team_4z = assign_values_to_team4(team2)

    input_data = pd.concat([team_3z,team_4z])
    return input_data

#get the input data and preprocess the data using the loaded data processing Pipeline,
#and predict the match result probabilities using predict_proba function, and return a dataframe with the probabilites.
def predict_match_result2(team3 ,team4):

    input_d = map_inputs_to_data2(team3 , team4)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test = model2.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes2,data=np.round(preds_test,3))
    results_df.rename(columns={0:'Draw Probability',1:'{} wins Probability'.format(team3),2:'{} wins Probability'.format(team4)},inplace=True)
    return results_df

#Predict function for the final match result prediction
def predict_match_result_goals(team3 ,team4):

    input_d = map_inputs_to_data2(team3 , team4)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test = model3.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes3,data=np.round(preds_test,4))
    draw_df = results_df[[x for x in results_df.columns if (int(x[0]) == int(x[2]))]]
    home_w_df = results_df[[x for x in results_df.columns if (int(x[0]) > int(x[2]))]]
    away_w_df = results_df[[x for x in results_df.columns if (int(x[0]) < int(x[2]))]]

    return draw_df,home_w_df,away_w_df
    
	
if __name__=='__main__':
	main()

