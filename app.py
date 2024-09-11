import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\bhupi\A VS CODE\1 Machine Learning\House Predictions\houseprice_prediction_model.pkl', 'rb'))

# Set the title of the Streamlit app
st.title("House_Price_Prediction_App")

# Add a brief description
st.write("This app predicts the house price based on Squarefeet using a simple linear regression model.")

# Add input widget for user to enter years of experience
Squarefeet = st.number_input("Enter Squarefeet:", min_value=500.000, max_value=1000.0,step=0.5)

# When the button is clicked, make predictions
if st.button("Predict House Price"):
    # Make a prediction using the trained model
    Squarefeet_input = np.array([[Squarefeet]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(Squarefeet_input)
   
   
    # Display the result
    st.success(f"The predicted price for the house for {Squarefeet} Squarefeet is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of House Prices and Square Feet.")