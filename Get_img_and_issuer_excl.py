from os import listdir
from os.path import isfile, join
import requests
import re

def get_issuer_exclusivity(terms):
    issuer_dict = {'visa':'visa', 'master':'mastercard|master card', 'mx':'americanexpress|american express', 'up':'unionpay|union pay'}
    issuer_count = {}
    for key in issuer_dict.keys():
        issuer_count[key] = len(re.findall(issuer_dict[key], terms.lower()))
    if max(issuer_count.values())==0:
        return 'all'
    else:
        return  max(issuer_count, key=issuer_count.get)

def get_image(img_url, img_set, dir_path, img_name):
    try:
        img_path = dir_path+ img_name + re.findall(r'(\.jpg|\.png)', img_url)[0]
    except:
        img_path = dir_path+ img_name + '.jpg'
    if img_name not in img_set:
#         img_res = requests.get(img_url)
#         with open(img_path, 'wb') as handle:
#             handle.write(img_res.content)
        img_set.add(img_name)
    return img_path

