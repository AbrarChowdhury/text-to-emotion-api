from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


from transformers import pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)
# print(prediction)

# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
# model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")


# def do_something(text): 
#     prediction = text
#     # Perform your processing on the text here
#     processed_text =  prediction # Example: Convert the text to uppercase
#     return processed_text

# @app.route('/')
# def hello_world():
#     return jsonify(message='Hello, World!')

@app.route('/', methods=['POST'])
def process_text():
    try:
        # Get the text from the request body
        data = request.get_json()
        input_text = data['text']

        # Call the do_something function to process the text
        emotions = classifier(input_text, )

        # Return the processed text as JSON response
        return jsonify(emotions)
    except Exception as e:
        return jsonify(error=str(e)), 400
    
if __name__ == '__main__':
    app.run(debug=True)