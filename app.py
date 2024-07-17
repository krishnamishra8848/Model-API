from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'iris_decision_tree_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return '''
        <h1>Iris Flower Prediction</h1>
        <form action="/predict" method="get">
            <label for="sepal_length">Sepal Length:</label>
            <input type="text" id="sepal_length" name="sepal_length" required><br><br>
            
            <label for="sepal_width">Sepal Width:</label>
            <input type="text" id="sepal_width" name="sepal_width" required><br><br>
            
            <label for="petal_length">Petal Length:</label>
            <input type="text" id="petal_length" name="petal_length" required><br><br>
            
            <label for="petal_width">Petal Width:</label>
            <input type="text" id="petal_width" name="petal_width" required><br><br>
            
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['GET'])
def predict():
    # Get the features from the query parameters
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Map the predicted class index to the actual class name
    target_names = ['setosa', 'versicolor', 'virginica']
    predicted_class = target_names[prediction[0]]
    
    # Return the prediction as JSON
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
