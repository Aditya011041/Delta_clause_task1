import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

def preprocess_input(sepal_length, sepal_width, petal_length, petal_width):
    # Perform preprocessing on the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Standardize the input features using StandardScaler
    scaler = StandardScaler()
    input_data_standardized = scaler.fit_transform(input_data)
    
    return input_data_standardized

def predict_iris_species(model, input_data):
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    # Convert predictions to class labels
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# Load the trained Iris classification model
model = load_model('iris_model.h5')  # Replace 'iris_model.h5' with the actual model file path

# Example input for prediction
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

# Preprocess the input data
input_data = preprocess_input(sepal_length, sepal_width, petal_length, petal_width)

# Make predictions
predicted_class = predict_iris_species(model, input_data)

# Map the predicted class index to the actual species
class_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species = class_mapping[predicted_class[0]]

# Print the predicted species
print(f"The predicted species is: {predicted_species}")
