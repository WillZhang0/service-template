from flask import Flask, request, after_this_request
from flask_restful import Resource, Api
import service
import os, shutil
import time
import Datashop

app = Flask(__name__)
api = Api(app)

class predict(Resource):
    @staticmethod
    def post():
        try:
            start = time.time()
            input_dict = request.get_json()

            print(input_dict)

            # Phase 1:  download userinput data and save in "tmp" folder
            jobID , inputdata = Datashop.phase_1(input_dict)

            """
            call the service below  
            follow service documentation for service output format
            """
            # Phase 2:  call the service
            service_results = service.run(jobID) # returns list of all the results

            # Phase 3:  save the results and update job status
            insightsS3Link = Datashop.save_results("filename", service_results[0], datatype="str") #specify the dtype of result (image,csv,graph,json,str,int)

            duration = time.time() - start
            #Update job status
            insights_payload = Datashop.updateJob(jobID,insightsS3Link,duration) #insightsS3Link can be a list of links [insightsS3Link1 ,insightsS3Link2 , insightsS3Link3]

            return {"result": "success","duration":duration,"insightFileURL":insights_payload}
                        
        except Exception as e:
            print("Error",e)
            #updating job with FAILED status.
            try:
                duration = time.time() - start;
                Datashop.updateJob(jobID,None, duration ,error=str(e))
                return {"result": "failed","duration":duration, "insightFileURL":str(e)}

            except Exception as e:
                return {"result": "update failed","duration": None, "insightFileURL":str(e)}

api.add_resource(predict,'/predict')

if __name__ == '__main__':
    app.run(debug=True)


