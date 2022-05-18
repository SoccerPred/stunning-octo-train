Build and deploy

Command to build the application. PLease remeber to change the project name and application name

gcloud builds submit --tag gcr.io/wc-predict/WorldCup-Prediction  --project=wc-predict
  
Command to deploy the application
  
gcloud run deploy --image gcr.io/wc-predict/WorldCup-Prediction --platform managed  --project=wc-predict --allow-unauthenticated
