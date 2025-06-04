import requests

def process(id, audio_path):
    # 1. Transcript
    resp = requests.post('http://localhost:5001/transcript', data={'id': id})  # , files={'audio': open(audio_path, 'rb')}
    transcript = resp.json()['transcript']

    # 2. Analysis
    resp = requests.post('http://localhost:5002/analyze', json={'id': id, 'transcript': transcript})
    summary = resp.json()['summary']

    # 3. Embedding
    resp = requests.post('http://localhost:5003/embed', json={'id': id, 'summary': summary})
    chunks = resp.json()['chunks']
    embeddings = resp.json()['embeddings']

    # 4. Store
    resp = requests.post('http://localhost:5004/store', json={'id': id, 'chunks': chunks, 'embeddings': embeddings})
    print(resp.json())

if __name__ == '__main__':
    process('test_id', 'test.m4a')
