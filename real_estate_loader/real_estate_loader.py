import json
import requests
import xmltodict as xmltodict
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from numpy import radians, cos, sin, sqrt
from numpy import arcsin as asin
import numpy as np
from moviepy.editor import *
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from training_data import json_converter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import calinski_harabasz_score


# from nylon import Polymer
# nylon_object = Polymer('/content/diabetes.csv')

# https://electrek.co/2023/01/10/tesla-applies-massive-million-expansion-gigafactory-texas/
# https://www.cnbc.com/2023/01/10/tesla-plans-to-spend-more-than-770-million-on-texas-factory-expansion.html
# https://www.theverge.com/2023/1/11/23549895/tesla-texas-factory-expansion-gigafactory


# Opening JSON file
# f = open("/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/zillow_austin.json")
# # returns JSON object as a dictionary
# z_json = json.load(f)
# f.close()
#
# _f = open(
#     "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/final_super_chargers.json")
# sc_json = json.load(_f)
# _f.close()
#
# __f = open(
#     "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/sc_geo_dict.json")
# geo_loc_json = json.load(__f)
# __f.close()
#
___f = open(
    "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/tx_sc_dist_from_zillow.json")
tgf_dist_from_zillow = json.load(___f)
___f.close()

____f = open(
    "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/tx_tgf_dist_from_zillow.json")
sc_dist_from_zillow = json.load(____f)
____f.close()

_____f = open(
    "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/filteredTX.json")
fz_json = json.load(_____f)
_____f.close()


# # # ZILLOW API ############################## Search Zillow API for current listings
# # ChatGPT Prompt: Write me an api call to pull a current stream of for
# # sale real estate postings in austin, texas, USA using python
# #
# # API endpoint
# url = "https://zillow56.p.rapidapi.com/search"
# querystring = {"location": "austin, tx",
#                "rentzestimate": "true",
#                "bedrooms": "1+",
#                "bathrooms": "1+",
#                "squareFeet": "1+",
#                "price": "1+",
#                # "latitude": "0+",
#                "livingArea": "0+",
#                "latitude": {"$gt": 0},
#                "lotAreaValue": {"$gt": 100}}
# headers = {
#     "X-RapidAPI-Key": "d6492acaa1msh0f4338d49e96062p1d120djsnd28f8e342831",
#     "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
# }
# response = requests.request("GET", url, headers=headers, params=querystring)
#
# # Check if the API call was successful
# if response.status_code == 200:
#     print(response.text)
# else:
#     print("API call failed with status code:", response.status_code)
# print(response.json())

# z_json = response.json()

# # Serializing json
# json_object = json.dumps(z_json, indent=4)
# # Writing to sample.json
# with open("sample01.json", "w") as outfile:
#     outfile.write(json_object)

# Check if the required keys are present in the property data
def load_and_filter_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    filtered_data = []
    required_keys = ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude']

    for obj in data['results']:
        if all(key in obj for key in required_keys) and ('lotAreaUnit' in obj or 'livingArea' in obj):
            if all(int(obj[val]) != 0 for val in required_keys) and (obj not in filtered_data):
                filtered_data.append(obj)

    # Serializing json
    json_object = json.dumps(filtered_data, indent=4)
    # Writing to sample.json
    with open("filtered01.json", "w") as outfile:
        outfile.write(json_object)

    return filtered_data

i = "/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/zillow_texas.json"
z_json = load_and_filter_json(i)




# # ZILLOW API ###############################

# # Google Maps API ###############################
# GEO_DICT = {}
# for sc in sc_json:
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     # https: // developers.google.com / maps / billing - and -pricing / pricing  # distance-matrix
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     # '        https://www.google.com/maps/embed/v1/MAP_MODE?key=AIzaSyBj2td7I4g-fF9ZV_GqrNALth0_ESoKcTA'
#     GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
#     params = {
#         'address': (sc_json[sc]['address'] + ' ,' + sc_json[sc]['city_state_zipcode'])[:-1],
#         'sensor': 'false',
#         'region': 'texas',
#         'key': 'AIzaSyBj2td7I4g-fF9ZV_GqrNALth0_ESoKcTA'
#     }
#     # Do the request and get the response data
#     req = requests.get(GOOGLE_MAPS_API_URL, params=params)
#     res = req.json()
#     # Use the first result
#     lat_lng = res['results'][0]['geometry']['location']
#     GEO_DICT[sc] = lat_lng
#     # creating json file
#     out_file = open("geo_dict.json", "w")
#     json.dump(GEO_DICT, out_file, indent=4)
#     out_file.close()
# # Google Maps API ###############################

# # Calculate Distances ###############################

# https://geohack.toolforge.org/geohack.php?pagename=Gigafactory_Texas&params=30.22_N_97.62_W_region:US-TX_type:landmark_dim:2000&title=Giga+Texas%3A+Tesla+Colorado+River+Project+%2F+Cybertruck+factory
# https://en.wikipedia.org/wiki/List_of_Tesla_factories

# # Gigafactory Texas
# tgf_geoloc = (30.22, -97.62)
#
#
# sc_dist_from_zillow = {}
# tgf_dist_from_zillow = {}
# sc_dist_avg = []
# tgf_dist_km = []
# house_ids = []
# for j in z_json:
#     # Loading the lat-long data for Kolkata & Delhi
#     try:
#         z_lat = j["latitude"]
#         z_long = j["longitude"]
#     except:
#         print()
#
#     z_dist_km = []
#     for i in geo_loc_json:
#         z_geodist = geodesic((z_lat, z_long), (geo_loc_json[i]['lat'], geo_loc_json[i]['lng'])).km
#         z_dist_km.append(z_geodist)
#     tgf_dist_km.append(geodesic((z_lat, z_long), (tgf_geoloc[0], tgf_geoloc[1])).km)
#     sc_dist_avg.append(np.average(z_dist_km))
#     house_ids.append(j['zpid'])
# keys = list(house_ids)
#
# sc_dist_from_zillow = []
# tgf_dist_from_zillow = []
# for i_tfg in tgf_dist_km:
#     sc_dist_from_zillow.append(i_tfg)
# for i_sc in sc_dist_avg:
#     tgf_dist_from_zillow.append(i_sc)
#
# # # creating json files
# out_file = open("tx_sc_dist_from_zillow.json", "w")
# json.dump(sc_dist_from_zillow, out_file, indent=4)
# out_file.close()
# #
# out_file = open("tx_tgf_dist_from_zillow.json", "w")
# json.dump(tgf_dist_from_zillow, out_file, indent=4)
# out_file.close()

# Calculate Distances ###############################


#################### Train a model on housing Data

# write python code for a machine learning model using
# pytorch for an m1 mac optimised trainer using MPS gpu
# parallel training that will learn to predict when
# housing listings, prices and sales are affected by
# the tesla mega factorys in travis country austin texas

# Load housing data
data = {"tgf": [], "sc": [], "bathrooms": [], "bedrooms": [], "price": [], "sqft": []}

for zj in z_json:
    data["bedrooms"].append(zj["bedrooms"])
    data["bathrooms"].append(zj["bathrooms"])
    data["price"].append(zj["price"])
    try:
        data["sqft"].append(zj["livingArea"])
    except:
        print()
data["tgf"] = json_converter.flatten_list(tgf_dist_from_zillow)
data["sc"] = json_converter.flatten_list(sc_dist_from_zillow)

# Convert data to tensors for use with PyTorch
# X = torch.tensor([data['tgf'], data['sc'], data['bedrooms'], data['bathrooms'], data['sqft'], data['price']], dtype=torch.float32)
# y = torch.tensor(data['price'], dtype=torch.float32)

# # Split the data into training and testing sets
# X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
data_tensors = [torch.tensor(data[key], dtype=torch.float32) for key in data.keys()]

tgf = torch.tensor(data['tgf'], dtype=torch.float32)
sc = torch.tensor(data['sc'], dtype=torch.float32)
bathrooms = torch.tensor(data['bathrooms'], dtype=torch.float32)
bedrooms = torch.tensor(data['bedrooms'], dtype=torch.float32)
price = torch.tensor(data['price'], dtype=torch.float32)
sqft = torch.tensor(data['sqft'], dtype=torch.float32)

# Concatenate the tensors into a single tensor
# Concatenate the tensors along the first dimension
z_input_data = torch.stack([tgf, sc, bathrooms, bedrooms, price, sqft], dim=1)

# data_tensor = torch.cat(data_tensors, dim=6)
#
# # Convert the tensor to a PyTorch TensorDataset
# dataset = TensorDataset(data_tensor)

# Create the dataloader
z_dataloader = DataLoader(z_input_data, batch_size=32, shuffle=True, pin_memory=True)

class RealEstateClustering(pl.LightningModule):
    def __init__(self, n_clusters, input_dim):
        super(RealEstateClustering, self).__init__()
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.centroids = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        x = x.unsqueeze(1)
        distances = ((x - self.centroids) ** 2).sum(2)
        _, cluster_assignments = distances.min(1)
        return cluster_assignments

    def loss(self, x, cluster_assignments):
        cluster_assignments = cluster_assignments.unsqueeze(1)
        distances = ((x - self.centroids) ** 2).sum(2)
        cluster_distances = distances.gather(1, cluster_assignments).squeeze(1)
        return cluster_distances.mean()

    def training_step(self, batch, batch_idx):
        x = batch
        cluster_assignments = self.forward(x)
        loss = self.loss(x, cluster_assignments)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        cluster_assignments = self.forward(x)
        loss = self.loss(x, cluster_assignments)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        cluster_assignments = self.forward(self.val_data)
        score = calinski_harabasz_score(self.val_data.numpy(), cluster_assignments.numpy())
        logs = {'val_loss': avg_loss, 'val_score': score}
        return {'val_loss': avg_loss, 'log': logs}

# Initialize the model and move it to GPU for parallel training
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

device = torch.device("mps" if mps_available else "cpu")
model = RealEstateClustering(n_clusters=3, input_dim=6)
model.to(device)

# Train the model on a GPU if available
trainer = pl.Trainer(max_epochs=100, gpus=1, devices='mps', accelerator='mps')
trainer.fit(model, train_dataloaders=z_dataloader)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.to(device)
# Save the model
torch.save(model.state_dict(), 'model.pt')

# # ################################# Load Texts into Video
# # Here is an example of how you can add text to a video with
# # background music using the moviepy library in Python:
#
# # Load video clip
# video = VideoFileClip("video.mp4")
#
# # Load background music
# audio = AudioFileClip("music.mp3")
#
# # Add background music to video
# video = video.set_audio(audio)
#
# # Add text to video
# txt_clip = ( TextClip("Your Text Here", fontsize=70, color='white')
#              .set_position('center')
#              .set_duration(video.duration) )
#
# final = CompositeVideoClip([video, txt_clip])
#
# # Write final video to file
# final.write_videofile("final_video.mp4")
#
# #############
