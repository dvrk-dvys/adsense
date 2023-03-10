# Python program to convert text
# file to JSON
import json
from itertools import islice, zip_longest
from numpy import radians, cos, sin, sqrt
from numpy import arcsin as asin
import numpy as np

# the file to be converted
filename = '_super_chargers'
# filename = '_amzn_fullfillment_centers'

# resultant dictionary
dict1 = {}

# fields in the sample file
keys = ['address', 'city_state_zipcode', 'roadside_assistance_phone']
# keys = ['state', 'building', 'type', 'address']

with open("/Users/jordanharris/Code/PycharmProjects/adsense-bot/real_estate_loader/training_data/" + filename + ".json") as fh:
    # count variable for employee id creation
    dict = {}
    lines = fh.readlines()
fh.close()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def grouper(n: str, iterable: object, fillvalue: None = None) -> object:
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

N = len(keys)
cnt = 00

cities = []
data = []
for i in lines:
    i = i.split('\t')
    res = any(chr.isdigit() for chr in i[0])

    if res == False:
        cities.append(i)
        continue
    data.append(i)


pre_grp = list(grouper(N,  data, fillvalue=None))
grp = []
cnt = 00

for j in pre_grp:
    _j = flatten_list(j)
    prep = {keys[j]: _j[j].strip() for j in range(len(keys))}
    pre = prep['roadside_assistance_phone'].split(':')[1].strip()
    prep['roadside_assistance_phone'] = pre
    dict[cnt] = prep
    cnt += 1




# creating json file
out_file = open("final" + filename + ".json", "w")
json.dump(dict, out_file, indent=4)
out_file.close()
