# ccp-customer-churn-prediction
describe churn

Customer Churn Prediction

The Customer Churn Prediction Project is an end-to-end machine learning solution designed to predict whether a customer is likely to leave a company’s services. The project uses the Telco Customer Churn dataset, which contains various customer attributes such as contract type, payment method, tenure, and monthly charges. The main goal is to analyze patterns in customer behavior and identify the key factors that contribute to churn, allowing businesses to take preventive actions.

This project is built using Python, Scikit-learn, Pandas, and Streamlit. It follows a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment through an interactive web interface. The trained model used in this project is a Random Forest Classifier, which provides a balance between accuracy and interpretability.

To begin, the dataset is cleaned by handling missing values, encoding categorical features, and normalizing numerical variables. Exploratory data analysis is then performed to visualize important trends such as churn rate distribution, relationships between contract type and churn, and the correlation between customer tenure and churn probability. These insights help understand customer behavior before moving into model training.

The model training process involves splitting the dataset into training and testing sets, fitting the Random Forest model, and evaluating it using key metrics such as accuracy, precision, recall, F1-score, and ROC AUC. The trained model and related metadata are saved in the models folder for reuse. On average, the model achieves an accuracy of around 80–85% and an ROC AUC score of approximately 0.83, depending on data randomness.

A simple Streamlit web application is included to allow users to interact with the model in real time. Through the app, users can input customer information manually or upload a CSV file containing multiple records to predict churn for multiple customers. The app displays whether a customer is likely to churn along with the predicted probability and also allows exporting results as a CSV file.

The project is structured into several directories, including data for the dataset, src for Python source files (data preparation, EDA, and model training), models for saved models, and app for the Streamlit interface. The required dependencies are listed in requirements.txt, making the setup straightforward.

To run the project, users need to create a virtual environment, install dependencies, add the dataset to the data folder, and train the model by executing the training script. After training, the Streamlit app can be launched locally to explore predictions through a browser interface. The app runs by default at http://localhost:8501
.

This project can be further improved by experimenting with different algorithms such as XGBoost or Logistic Regression, fine-tuning hyperparameters using GridSearchCV, adding feature importance visualization, or deploying the Streamlit app online using Streamlit Cloud or Heroku.
