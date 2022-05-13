from flask import Flask, request, after_this_request
from flask_restful import Resource, Api
from main import main

app = Flask(__name__)
api = Api(app)

class predict(Resource):
    @staticmethod
    def post():
        #try:
        input_dict = request.get_json()
        insights_payload, duration = main(input_dict)
        #return {"result": "success", "duration": duration, "insightFileURL": insights_payload}
        #except Exception as e:
            ## updating job with FAILED status.
            #return {"result": "update failed","duration": None, "insightFileURL":str(e)}

api.add_resource(predict,'/predict')

if __name__ == '__main__':
    app.run()
