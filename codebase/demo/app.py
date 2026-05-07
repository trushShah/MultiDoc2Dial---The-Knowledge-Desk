from flask import Flask, request, jsonify, send_from_directory
import os

# Import the RAG pipeline
import sys; import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.generator import generate_answer

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    query = data['query'].strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        # Call the refactored generate_answer which returns a dict
        result = generate_answer(query)
        return jsonify(result)
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=False, debug=False)
        print("Flask app exited cleanly.")
    except Exception as e:
        print(f"Exception caught in app.run: {e}")
