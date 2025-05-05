from flask import Flask, request, render_template
import joblib
import numpy as np


flask_app = Flask(__name__)


tanaman = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
           'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
           'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
           'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']


try:
    model = joblib.load("model_rf.pkl")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Endpoint for the home page
@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/prediction", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("index.html", prediction_text="Error: Model could not be loaded.")

        
        float_features = [float(x) for x in request.form.values()]
        
        
        if len(float_features) != 7:
            return render_template("index.html", prediction_text="Error: Exactly 7 features are required.")

      
        features = np.array(float_features).reshape(1, -1)

      
        prediction = model.predict(features)
        pred_label = int(prediction[0])
        
       
        if 0 <= pred_label < len(tanaman):
            predicted_crop = tanaman[pred_label]
            return render_template("index.html", prediction_text=f"The Predicted Crop is: {predicted_crop}")
        else:
            return render_template("index.html", prediction_text="Error: Invalid prediction index.")

    except ValueError as ve:
        return render_template("index.html", prediction_text="Error: Invalid input. Please enter numeric values.")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    flask_app.run(debug=True)
