from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed():
    id = request.json['id']
    summary = request.json['summary']
    # TODO: 这里用占位符
    chunks = [summary[i:i+10] for i in range(0, len(summary), 10)]
    embeddings = [[0.1]*5 for _ in chunks]  # 假设每个embedding是5维
    return jsonify({'id': id, 'chunks': chunks, 'embeddings': embeddings})

if __name__ == '__main__':
    app.run(port=5003) 