import nltk
from flask import Flask, request, render_template, url_for
import json
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
import random
app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = load_model('model.h5')
lemma = WordNetLemmatizer()
intents = json.load(open('intents.json'))
words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('class.pkl', 'rb'))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemma.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    cltn = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                cltn[i] = 1
                if show_details:
                    print(f"Found '{w}' in bag")
    return cltn

def predict_class(sentence, model):
    l = bow(sentence, words, show_details=False)
    res = model.predict(np.array([l]))[0]

    ERROR_THRESHOLD = 0.25
    results = [(i, j) for i, j in enumerate(res) if j > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[k[0]], "probability": str(k[1])} for k in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
orders={
            "12345": "Shipped - Estimated delivery: 2025-01-25",
            "67890": "Processing - Expected to ship within 2 days",
            "54321": "Delivered on 2025-01-20",
            "56822": "Refund Under Process"
        }


def get_order_status(order_id):
    return orders.get(order_id, "Order ID not found. Please check and try again.")
def chatbotResponse(msg):
    if msg.isdigit():
        order_id = ''.join(filter(str.isdigit, msg))
        if order_id:
            return get_order_status(order_id)
        else:
            return "Please provide a valid order ID."
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)

    return res
orders={
            "12345": "Shipped - Estimated delivery: 2025-01-25",
            "67890": "Processing - Expected to ship within 2 days",
            "54321": "Delivered on 2025-01-20",
            "56822": "Refund Under Process"
        }


@app.route('/')
def home():
    return render_template('chatbot.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = chatbotResponse(msg)
    return response

if __name__ == '__main__':
    app.run(debug=True)

