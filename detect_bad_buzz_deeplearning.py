import os
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)

STATIC_FOLDER = 'static/'
MODEL_FOLDER = STATIC_FOLDER + 'models/'

@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    global model
    #model = load_model(MODEL_FOLDER + 'bidirectional_lstm_with_return_sequences_on_embedded_heroku')
    model = load_model(MODEL_FOLDER + 'ffnn_on_count')
    print('[INFO] : Model loaded')


def predict(text_to_predict):
    # Prediction:
    y_test_pred_proba = model.predict([text_to_predict], batch_size=1, workers=4,)
    result= round(y_test_pred_proba[0][0],2)
    return result

# Home Page
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        text_to_predict = request.form["tweet_to_predict"]
        print(text_to_predict)
        pred_prob = predict(text_to_predict)

        if pred_prob > .5:
            label = 'Positif'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Négatif'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('index.html', text_to_predict=text_to_predict, label=label, accuracy=accuracy, predict=True)
    else:

        return render_template('index.html', predict=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host="0.0.0.0", port=port, debug=True)
