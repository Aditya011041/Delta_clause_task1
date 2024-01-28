import os
from flask import Flask, render_template, request, redirect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import jsonify

app = Flask(__name__)
app.secret_key = 'supersecretkey'

login_manager = LoginManager(app)
login_manager.login_view = '/'


class User(UserMixin):
    def get_id(self):
        return self.id

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

model = load_model('iris_model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Dummy user data (replace with your user database logic)
# users = {
#     "user1": {
#         "username": "user1",
#         "password": generate_password_hash("password1")
#     },
#     "user2": {
#         "username": "user2",
#         "password": generate_password_hash("password2")
#     }
# }

labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

def preprocess_input(sepal_length, sepal_width, petal_length, petal_width):
    # Preprocess the input for the Iris model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return input_data

def get_iris_result(input_data):
    # Make predictions using the Iris model
    predictions = model.predict(input_data)
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
@login_required
def iris_index():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = preprocess_input(sepal_length, sepal_width, petal_length, petal_width)
        predictions = get_iris_result(input_data)

        predicted_label = labels[np.argmax(predictions)]
        return render_template('index.html', predicted_label=predicted_label)

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            input_data = preprocess_input(sepal_length, sepal_width, petal_length, petal_width)
            predictions = get_iris_result(input_data)

            predicted_label = labels[np.argmax(predictions)]

            # Debugging: Print the predicted label and response content
            print("Predicted Label:", predicted_label)
            print("Response Content:", jsonify({'predicted_label': predicted_label}))

            return jsonify({'predicted_label': predicted_label})

    except Exception as e:
        print("Error occurred:", e)

    return jsonify({'error': 'Invalid request'})



if __name__ == '__main__':
    app.run(debug=True)
