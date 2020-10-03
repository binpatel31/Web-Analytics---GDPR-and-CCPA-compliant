from flask import Flask, jsonify, abort, make_response, request, render_template, url_for
import json

app = Flask(__name__)


'''
POST implementation: creation of new todo
'''
@app.route('/')
def create_todo_note():
    return render_template("webpage.html")

@app.route("/json_data", methods=["POST","GET"])
def get_json_data():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
    return '', 200


if __name__ == '__main__':
    app.run(debug = True)
