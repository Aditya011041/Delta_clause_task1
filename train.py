import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from a CSV file
iris_data = pd.read_csv("C:\Desktop\iris\Dataset\iris.csv")  

# Display the first few rows of the dataset
print(iris_data.head())

# Extract features and labels
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Convert labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Model architecture
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_val, y_val))

# Plotting the training history
sns.set_theme()
sns.set_context("poster")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model
model.save("iris_model.h5")
