from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)
DATA_DIR = '../data'

@app.route('/store', methods=['POST'])
def store():
    id = request.json['id']
    chunks = request.json['chunks']
    embeddings = request.json['embeddings']
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f'{id}.json'), 'w') as f:
        json.dump({'id': id, 'chunks': chunks, 'embeddings': embeddings}, f)
    return jsonify({'status': 'success', 'id': id})

if __name__ == '__main__':
    app.run(port=5004)
