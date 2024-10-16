import time
from ChemAgent import OpenAIAgent
from flask import request, jsonify, Flask, json, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/chat/setuserN', methods=['POST'])
def setuserN():
    data = json.loads(request.get_data())
    inputt = data["userinfo"]
    chat = OpenAIAgent()
    print("this is input: ", inputt, end='\n')
    result = chat.run(inputt)

    print("this is result", result)
    return jsonify({"botanswer": result, "step": 1, "message": "true"})

if __name__ == "__main__":
    app.run()
