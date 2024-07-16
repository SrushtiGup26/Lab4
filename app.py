from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
rf_model = joblib.load('fish_species_model.pkl')

# Flask routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_species', methods=['POST'])
def predict_species_route():
    if request.method == 'POST':
        # Retrieve form data
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])

        # Predict species using the loaded model
        prediction = rf_model.predict(
            [[length1, length2, length3, height, width]])[0]

        # Assuming you have a mapping of class indices to species names
        species_mapping = {
            0: 'Bream',
            1: 'Roach',
            2: 'Perch',
            3: 'Pike',
            4: 'Smelt',
            5: 'Parkki'
        }

        predicted_species = species_mapping.get(prediction, 'Unknown')

        return render_template('index.html', prediction=predicted_species)


if __name__ == '__main__':
    app.run(debug=True)
