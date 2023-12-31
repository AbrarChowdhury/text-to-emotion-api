
# ? pip install tensorflow flask flask_cors transformers
import time
import json
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins=["https://virtyousandbox.com:8444","http://localhost:8444"])
from transformers import pipeline
# conversation_classifier = pipeline("conversational", model="facebook/blenderbot-400M-distill")
emotions_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('emotions')
def handle_message(message):
    emotions = emotions_classifier(message['text'])
    socketio.emit(message['sessionId'], emotions)

@app.route('/', methods=['POST'])
def process_text():
    try:
        # Get the text from the request body
        data = request.get_json()
        input_text = data['text']
        tic = time.perf_counter()
        # Call the do_something function to process the text
        emotions = emotions_classifier(input_text)
        toc = time.perf_counter()
        print(f"Time -> {toc-tic:0.4f} seconds")

        # Return the processed text as JSON response
        return jsonify(emotions)
    except Exception as e:
        return jsonify(error=str(e)), 400
    
if __name__ == '__main__':
    certfile = '/etc/letsencrypt/live/virtyousandbox.com/fullchain.pem'
    keyfile = '/etc/letsencrypt/live/virtyousandbox.com/privkey.pem'
    socketio.run(app, debug=True, host='0.0.0.0', port=5444, ssl_context=(certfile, keyfile))
    # socketio.run(app, debug=True, host='0.0.0.0', port=5444)

