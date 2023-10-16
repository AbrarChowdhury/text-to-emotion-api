
# ? pip install tensorflow flask flask_cors transformers

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

from transformers import pipeline
# conversation_classifier = pipeline("conversational", model="facebook/blenderbot-400M-distill")
emotions_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)



@app.route('/', methods=['POST'])
def process_text():
    try:
        # Get the text from the request body
        data = request.get_json()
        input_text = data['text']

        # Call the do_something function to process the text
        emotions = emotions_classifier(input_text)

        # Return the processed text as JSON response
        return jsonify(emotions)
    except Exception as e:
        return jsonify(error=str(e)), 400
    
if __name__ == '__main__':
    app.run(debug=True)
