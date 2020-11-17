from flask import Flask, jsonify, abort, make_response, request, render_template, url_for
import json
import requests
from elasticsearch import Elasticsearch
import datetime

app = Flask(__name__)


'''
POST implementation: creation of new user record
'''
@app.route('/')
def home_page():
    return render_template("webpage.html")

@app.route("/json_data", methods=["POST","GET"])
def get_json_data():
    if request.method == 'POST':
        data = request.get_json()
        ### get city, state and country name
        url = "https://api.ipgeolocation.io/ipgeo?apiKey=2b1ee37501e64754b85f704fab4a5b82&ip="+data["ip"]
        resp = requests.get(url=url)
        info = resp.json()
       
        ### ==== step 1: using data to classify user as unique or not and get ID if it is
        platform = data["platformmodel"].lower()
        OS = data["os"].split("|")[0].lower()
        timezone = int(data["timezone"])
        user_agent = data["user_agent"].lower()
        browser = ""
        browser_version = ""
        try:
            browser, browser_version = [i.lower() for i in data["browser"].split()]
        except:
            browser = data["browser"].lower()
        channel, width, height = [ int(i) for i in  data["resolution"].split("|")]
        vendor = data["vendor"].lower()
        language = data["language"].lower()
        print(platform, "===", OS,"===", timezone, "===",user_agent,"===", browser,"===", browser_version, "===",channel, "===",width,"===", height, "===",vendor,"===", language)
        ### step 2 call model to get userID
        user_ID = 1 ### temp 



        ###
        data["country"] = info["country_name"]
        data["city"] = info["city"]
        data["state"] = info["state_prov"]
        data["time"] = datetime.datetime.now().isoformat()#strftime("%Y-%m-%d %H:%M:%S").isoformat()
        
        del data["ip"]
        print("********-------------*****")
        print("Data to send is ", data)
        print("*******--------------*****")
        es = Elasticsearch([{'host':'localhost','port':9200}])
        res = es.index(index='my-index-000001', body=data)
    return '', 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug = True)
