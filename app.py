import sys
import os
from flask import Flask, request, jsonify, render_template
from ollama import Client

# Add the UI_Enabled directory to the Python path
sys.path.append(os.path.dirname(__file__))

import styles  # Import the styles module

app = Flask(__name__)

# Ollama API setup
client = Client(host='http://localhost:11434')
conversation_history = []
max_context_size = 50

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('content')
    if user_input:
        conversation_history.append({"role": "user", "content": user_input})
        prompt = get_prompt()
        full_response = generate_response(prompt)
        conversation_history.append({"role": "assistant", "content": full_response})
        return jsonify({"response": full_response})
    return jsonify({"error": "No input provided"}), 400

def get_prompt():
    """Constructs the prompt with conversation history."""
    system_message = "You are a professional assistant. Respond formally."
    prompt = f"{system_message}\n"
    for entry in conversation_history[-max_context_size:]:
        if entry["role"] == "user":
            prompt += f"User: {entry['content']}\n"
        elif entry["role"] == "assistant":
            prompt += f"Assistant: {entry['content']}\n"
    return prompt

def generate_response(prompt):
    full_response = ""
    try:
        response = client.generate(
            model='llama3.2',
            prompt=prompt,
            stream=True,
            options={'temperature': 0.8}
        )
        for part in response:
            text_chunk = part['response']
            full_response += text_chunk
    except Exception as e:
        full_response = f"Error: {e}"
    return full_response

if __name__ == "__main__":
    app.run(debug=True)