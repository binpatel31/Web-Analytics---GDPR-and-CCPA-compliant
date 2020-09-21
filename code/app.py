from flask import Flask, jsonify, abort, make_response, request
import json
app = Flask(__name__)


'''
POST implementation: creation of new todo
'''
@app.route('/add', methods = ['POST'])
def create_todo_note():
        if not request:
                abort(400)
        data = json.loads(request.data)
        print(data)
        '''
        dic = {str(request.json['title']):[]}
        with open('data.json', 'r') as outfile:
                data = json.load(outfile)
                all_todos = data["all_todos"]
        
        all_todos.append(dic)
        with open('data.json', 'w') as outfile:
                json.dump({"all_todos":all_todos}, outfile)


        return jsonify({ 'List of all todos notes' : all_todos }),201
	

                json.dump({"all_todos":all_todos}, outfile)
        '''
        return jsonify({ 'Hello' : "world" }),201


if __name__ == '__main__':
	app.run(debug = True)
