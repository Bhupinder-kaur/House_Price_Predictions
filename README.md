# House_Price_Predictions

This project implements a Linear Regression model to predict house prices based on the living space (square footage) of the house. The code utilizes essential machine learning libraries such as scikit-learn, numpy, pandas, and matplotlib for model training, data processing, and visualization.

Project Structure
Data Preprocessing:

The dataset containing house prices and square footage is loaded using pandas.
The input feature (sqft_living) and the target variable (price) are extracted from the dataset and converted into numpy arrays.
The dataset is split into training and test sets using train_test_split.
Model Training:

A Linear Regression model from sklearn.linear_model is used to fit the training data (xtrain, ytrain).
After training, the model predicts prices for the test set (xtest), and visualizations are generated for both training and test datasets using matplotlib.
Model Evaluation:

The model's performance is evaluated using the R-squared (R²) score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Predictions for specific house sizes (e.g., 2525 sqft and 1000 sqft) are made and displayed.
Pickling the Model:

The trained model is saved to a file (houseprice_prediction_model.pkl) using the pickle library for future use or deployment.
Advanced Statistical Analysis:

Using statsmodels, the code provides an OLS (Ordinary Least Squares) summary for more detailed insights into the regression results.
Key Dependencies
numpy
pandas
matplotlib
scikit-learn
statsmodels
pickle
Visualizations
The project provides two sets of visualizations:

Training Dataset visualization showing the relationship between square footage and house prices.
Test Dataset visualization showing how well the model generalizes to unseen data.
Example Predictions
Predicted house price for 2525 sqft: $<predicted_value>
Predicted house price for 1000 sqft: $<predicted_value>
Metrics
R² Score: Measures the goodness of fit.
MSE and RMSE: Provide error metrics to evaluate prediction accuracy.
