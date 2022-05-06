import service
import time
import Datashop

def main(input_dict):

    start = time.time()

    # phase 1 Datashop initialization
    jobID = Datashop.initialize(input_dict)

    # phase 2 calling the service
    service_results = service.run(jobID)  # returns list of all the results

    # phase 2 save results
    insightsS3Link = Datashop.save_results("filename", service_results[0],
                                           datatype="image")  # specify the dtype of result (image,csv,graph,json,str,int)

    # calculate job duration
    duration = time.time() - start

    # phase 3 insightsS3Link can be a list of links [insightsS3Link1 ,insightsS3Link2 , insightsS3Link3]
    insights_payload = Datashop.updateJob(jobID, insightsS3Link, duration)

    return insights_payload , duration

if __name__ == "__main__":
    print("cannot run main funtion separately, please start the app.py")