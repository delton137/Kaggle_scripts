#!/usr/bin/env python
#Dan Elton, 2016
import numpy as np
import pandas as pd
import requests
import json
import pickle

#Import data using AirBnB's 'internal' API, which returns JSON

#sw_lat=40.833 
#sw_lng=-74.084 
#ne_lat=40.889
#ne_lng=-73.86

#json_url = "https://www.airbnb.com/search/search_results?page=1&source=map&airbnb_plus_only=false&sw_lat="+sw_lat+"&sw_lng="+sw_lng+"&ne_lat="+ne_lat+"&ne_lng="+ne_lng+"&search_by_map=true&location=Manhattan,+New+York,+NY,+United+States&ss_id=4nufl4i6"

listings = []

#each page contains 18 listings
for i in range(200):
    print(i)
    
    #NYC
    #json_url = "https://www.airbnb.com/search/search_results?page="+str(i)+"&source=map&airbnb_plus_only=false&search_by_map=true&location=Manhattan,+New+York,+NY,+United+States&ss_id=4nufl4i6"
    
    #San Francisco 
    #json_url = "https://www.airbnb.com/search/search_results?page="+str(i)+"&source=map&airbnb_plus_only=false&search_by_map=true&location=San+Francisco"
    
    #Los Angeles
    json_url = "https://www.airbnb.com/search/search_results?page="+str(i)+"&source=map&airbnb_plus_only=false&search_by_map=true&location=Los+Angeles"
    
    raw = requests.get(json_url).text # download the raw JSON

    data = json.loads(raw)           # parse JSON into a dict structure

    listings += data['results_json']['search_results']

num_listings = len(listings)

features = ['id','bedrooms','star_rating','person_capacity','lat','lng','reviews_count']
num_features = len(features)

#df = pd.DataFrame(np.zeros([num_listings,num_features]))

npdata = np.zeros([num_listings,num_features])
price = np.zeros(num_listings)

for i in range(num_listings):
    price[i] = listings[i]['pricing_quote']['rate']['amount'] 
        
    for j in range(num_features): 
        npdata[i][j] = listings[i]['listing'][features[j]]
 
pickle.dump((price, npdata),open("LA_3600_apts.pkl","wb"))