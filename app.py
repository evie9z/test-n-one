from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world!!s'

@app.route('/train')
def train_model():
    return "okjjk"


if __name__ == "__main__":
    app.run()