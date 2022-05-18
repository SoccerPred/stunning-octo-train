Build and deploy

Command to build the application. PLease remeber to change the project name and application name

gcloud builds submit --tag gcr.io/LZTechnologyLLC/WorldCup-Prediction  --project=LZTechnologyLLC
  
Command to deploy the application
  
gcloud run deploy --image gcr.io/LZTechnologyLLC/WorldCup-Prediction --platform managed  --project=LZTechnologyLLC --allow-unauthenticated
