import re
import time
import html
import pickle
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import CountVectorizer
import os

def pc_to_region(pc, df_region):
    """ Map the postal code to the corresponding region by using the region table
    :param pc: postal code
    :param df_region: dataframe of region
    :return: df with columns being ['PostalSector', 'PostalDistrict', 'PostalDistrictName', 'GeneralLocation']
            if no postal givent or no match, the function returns the same df but filled with '' 
    """
    if (pc == '') or (len(pc) != 6):
        return pd.DataFrame([['']*df_region.shape[1]], columns = df_region.columns).to_dict('records')[0]
    else:
        try:
            return df_region.loc[df_region['postal_sector'] == int(pc[:2])].to_dict('records')[0]
        except:
            return pd.DataFrame([['']*df_region.shape[1]], columns = df_region.columns).to_dict('records')[0]

def similar(a, b):
    """ Check the similar of two inputs
    :param a: first input
    :param b: second input
    :return: similarity of 2 inputs 
    """
    return SequenceMatcher(None, a, b).ratio()

def get_address(phone, name, gtypes_list, extract_status, df_region):
    """ Get the address by using Google API
    Note: Need to store the api key in a different path/folder/files
    :param phone: phone number [not using here!]
    :param name: name of merchant
    :param gtypes_list: list of google type [to check the correct of results]
    :param extract_status: status of extraction
    :param df_region: region table in dataframe
    :return: 
    """
    #file = open('keys.txt', 'r').readlines()
    #api_key = file[0].strip()
    #api_key = os.environ['gkey']
    api_key = AIzaSyDmS8E_1_3nkyjtw69sQyBb_aZjtvp98k8
    flag = 0
    res_phone = []
    res_name = []
    tags = re.compile('<.*?>')
    
    ### Retrieving all stores across island ###
    if isinstance(name, str) and extract_status == 'all':
        formated_name = name.replace(' ','%20')
        formated_name = re.sub(tags, '', formated_name)
        formated_name = html.unescape(formated_name)
        formated_name = formated_name.lower()
        try:
            res= requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=1.290849,103.844673&radius=20067.09&keyword='+ formated_name +'&key=' + api_key).json()
        except:
            try:
                res= requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=1.290849,103.844673&radius=20067.09&keyword='+ formated_name +'%20singapore' +'&key=' + api_key).json()
            except:
                res = {'results':[]}
            pass
        res_name = res['results']
        
        # Looping on all pages
        while 'next_page_token' in res.keys():
            res = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken=' + res['next_page_token'] + "&key="+api_key).json()
            res_name += res['results']
    
    # I kept the res_phone, but here it's empty so the following line doesn't do anything
    combined_res = res_phone + res_name
    # I get rid of duplicates
    combined_res = [i for n, i in enumerate(combined_res) if i not in combined_res[n + 1:]]

    ### Scoring loop to know if we retrieved the right stores
    if isinstance(name, str):
#         del_list = []
        for i in range(len(combined_res)):
            combined_res[i]['true_store'] = 0
            combined_res[i]['hamming_score'] = similar(name, combined_res[i]['name'])
            if combined_res[i]['hamming_score'] < 0.5:
                combined_res[i]['flag'] = 1
            elif combined_res[i]['hamming_score'] <0.1:
#                 del_list += [i]
                combined_res[i]['true_store'] = 1
            else:
                combined_res[i]['flag'] = 0
            
            if combined_res[i]['types']==[]:
                combined_res[i]['types'] = 1
            elif combined_res[i]['types'][0] not in gtypes_list:
                combined_res[i]['true_store'] = 2
            combined_res[i]['flag'] = max(combined_res[i]['flag'], combined_res[i]['true_store'])
    ### Postcode ###
    L = []
    flag_pc = 0
    for el in combined_res:
        postcode = get_postcode(el['geometry']['location']['lat'], el['geometry']['location']['lng'])
        if postcode == '':
            flag_pc = 2
            region = {
                        'postal_sector' : '',
                        'postal_district': '',
                        'postal_district_name': '',
                        'general_location': ''
            }
        else:
            region = pc_to_region(postcode, df_region)
        el['postal_code']=postcode
        el['postal_sector'] = region['postal_sector']
        el['postal_district'] = region['postal_district']
        el['postal_district_name'] = region['postal_district_name']
        el['general_location'] = region['general_location']
        el['flag'] = max(flag_pc, el['flag'])
            
    return combined_res

def get_postcode(lat, lng, postcode = ''):
    """ Get the postal code
        Note: It seems geopy blocks the connection after a certain use, 
        in case it does then use the following code to change the user_agent name.
        :param lat: latitude
        :param lng: longtiture
        :param postcode : None
        :return: postal code
    """
    if isinstance(postcode, str) and len(postcode) == 6:
        return postcode
    elif lat == np.nan or lng == np.nan or lat == '' or lng == '':
        return ''
    else:
        pass
    i = 0
    flag_ok = True
    while flag_ok:
        try:
            geolocator = Nominatim(user_agent="my-application" + str(i))
            location = geolocator.reverse(str(lat) + ',' + str(lng))
            flag_ok = False
        except:
            if i > 500:
                flag_ok = False
            else:
                pass
    try:
        if len(str(location.raw['address']['postcode']))==6:
            return location.raw['address']['postcode']
        elif len(str(location.raw['address']['postcode'])) == 5:
            return '0' + str(location.raw['address']['postcode'])
        else:
            return 'Online'
    except:
        return 'Online'

# def google_completion(name, extracted_address = '', gtypes_list=[], merchant_dict):
def google_completion(name, extracted_address, gtypes_list, merchant_dict, df_region):
    '''
        This fonction retrieves the data to complete the extracted deal. 
        /!\ For now merchant_dict and df_region are dealt as global parameters
        Input:  - name, which will be used for the query, str expected
                - address, as optional parameter which will then be matched to the corresponding address from the google api
        
        Output: DataFrame with the columns which are to be matched with the deal
    '''
    flag = 0
    
    ### Output dataframe's columns ###
    columns = ['latitude', 'longitude', 'merchant_id', 'store_id', 'google_name',
               'corr_name', 'corr_address', 'google_address', 'google_flag']
    
    if name not in merchant_dict.keys():
        address = get_address(None, name, gtypes_list, 'all', df_region)
        merchant_dict[name] = address
        for i,el in enumerate(merchant_dict[name]):
            el['merchant_id'] = len(merchant_dict.keys())
            el['store_id'] = i+1
        pickle.dump(merchant_dict, open('merchant_dict_clean.pickle', 'wb'))

    address = merchant_dict[name]
    ### Getting rid of addresss with flag >1
    address_tmp = []
    for el in address:
        if el['flag'] <=1:
            address_tmp += [el]
    address = address_tmp

    if address == []: 
        df_empty = pd.DataFrame([['']*len(columns)], columns = columns)
        df_empty.loc[0,'google_flag'] = 2
        return df_empty 
    else:
        df_address_tmp = pd.DataFrame(columns = columns) # + ['address_score'])
        for el in address:
            df_tmp = {}
            
            ### Filling google API results ###
            df_tmp['latitude'] = el['geometry']['location']['lat']
            df_tmp['longitude'] = el['geometry']['location']['lng']
            df_tmp['merchant_id'] = el['merchant_id']
            df_tmp['store_id'] = el['store_id']
            df_tmp['google_name'] = el['name']
            df_tmp['corr_name'] = el['hamming_score']
            df_tmp['google_address'] = el['vicinity']
            
            ### Filling region details ###
            df_tmp['postal_code'] = el['postal_code']
            df_tmp['postal_sector'] = el['postal_sector']
            df_tmp['postal_district'] = el['postal_district']
            df_tmp['postal_district_name'] = el['postal_district_name']
            df_tmp['general_location'] = el['general_location']
            df_tmp['google_flag'] = max(flag, el['flag'])
            
            ### Appending the current store to the list of stores
            df_address_tmp = df_address_tmp.append(pd.DataFrame(df_tmp, index=[0]), sort = True)
            
        df_address_tmp.reset_index(inplace = True, drop = True)
        
        ### Checking correlation with given address ###
        if isinstance(extracted_address, str) and extracted_address != '':
            df_address_tmp['corr_address'] = df_address_tmp['google_address'].apply(lambda x: similar(x, extracted_address))
            max_corr = max(df_address_tmp['corr_address'])
            df_address_tmp['true_address'] = np.where(df_address_tmp['corr_address']==max_corr, True,False)
            
            ### Flaging if low score ###
            if max_corr < 0.5:
                if len(df_address_tmp.loc[df_address_tmp['true_address']==True, 'google_flag'].values)==1:
                    df_address_tmp.loc[df_address_tmp['true_address']==True, 'google_flag'] =  max(df_address_tmp.loc[df_address_tmp['true_address']==True, 'google_flag'].values, 1)
                else:
                    df_address_tmp.loc[df_address_tmp['true_address']==True, 'google_flag'] = max(df_address_tmp.loc[df_address_tmp['true_address']==True, 'google_flag'].values[0], 1)
        else:
            df_address_tmp['corr_address'] = [0]*df_address_tmp.shape[0]
            df_address_tmp['true_address'] = [False]*df_address_tmp.shape[0]
        df_address_tmp['google_flag'] = np.where(df_address_tmp['google_flag'], np.nan, 0)
        return df_address_tmp

def completion_pipeline(df_raw, bank_name, filename, merchant_dict, df_region):
    df_raw.reset_index(inplace = True, drop = True) #reset index in case since we'll be looping using iloc
    
    
    columns = ['latitude', 'longitude', 'merchant_id', 'store_id', 'google_name',
                'corr_name', 'corr_address', 'true_address', 'postal_sector', 'postal_district', 'postal_district_name',
               'general_location', 'google_address']
    df_final = pd.DataFrame(columns = set(df_raw.columns.tolist() + columns))
    
    ### Creating deal_id ###
    country = 'sg'
    time.sleep(1)
    timestamp = int(time.time())
    seq = 0
    
    for i in tqdm(range(df_raw.shape[0])):
        deal = df_raw.iloc[i].to_frame().T
        
        ### Mapping bank categories to standard categories ###
        # start by retrieving bank's name
        if re.findall('_', deal['card_name'].item())!=[]:
            bank_name = deal['card_name'].item().split('_')[0]
        else:
            bank_name = 'dbs'
        
        ### Case if deal is an online deal ### 
        if deal.is_online.item() == True:
            deal['latitude'] = ''
            deal['longitude'] = ''
            deal['store_id'] = ''
            deal['merchant_id'] = ''
            deal['google_name'] = deal['merchant_name'].item()
            deal['corr_name'] = ''
            deal['corr_address'] = ''
            deal['google_address'] = ''
            deal['true_address'] = True
            deal['postal_code'] = ''
            deal = deal.merge(deal.postal_code.apply(lambda s: pd.Series(pc_to_region(s,df_region))), left_index=True, right_index=True)

        ### Case where we already have the store's location ###
        elif ('latitude' in deal.columns and deal.latitude.isnull().values.any() == False):
            deal['store_id'] = ''
            deal['merchant_id'] = ''
            deal['google_name'] = deal['merchant_name'].item()
            deal['corr_name'] = ''
            deal['corr_address'] = ''
            deal['google_address'] = deal['address'].item()
            deal['true_address'] = True
            
            ### Adding postcode ###
            deal['postal_code'] = deal[['latitude', 'longitude', 'postal_code']].apply(lambda x: get_postcode(x['latitude'], x['longitude'], x['postal_code']), axis = 1)
            
            ### Adding region information ###
            deal = deal.merge(deal.postal_code.apply(lambda s: pd.Series(pc_to_region(s,df_region))), left_index=True, right_index=True)
        
        ### Otherwise we use google API to complete the deal's information ###
        else:
            if 'latitude' in deal.columns:
                deal.drop('latitude', axis = 1, inplace = True)
                deal.drop('longitude', axis = 1, inplace = True)
            if 'postal_code' in deal.columns:
                deal.drop('postal_code', axis = 1, inplace = True)
            df_completion = google_completion(deal.iloc[0]['merchant_name'], deal.iloc[0]['address'], deal.iloc[0]['google_types'], merchant_dict, df_region)
            deal = pd.concat([pd.DataFrame([deal.values[0] for j in range(df_completion.shape[0])], columns = deal.columns),df_completion], axis = 1)
            deal['flag'] = deal[['flag', 'google_flag']].max(axis=1) # merging flags
            deal.drop('google_flag', axis = 1, inplace = True)
            deal.reset_index(drop = True, inplace = True)

        ### Appending to create the final dataframe ###
        df_final = df_final.append(deal, sort = True)
    df_final.reset_index(drop = True, inplace = True)
    ### Joining on card key
    card_table = pd.read_csv('card_table.csv')
    card_table = card_table.set_index('deal_handler')
    
    if 'card_name' in df_final.columns:
        df_final = df_final.rename({'card_name':'deal_handler'}, axis = 1)
    
    df_final = df_final.join(card_table, on ='deal_handler')
    
    ### Adding deal_id, uniaue per rows ###
    ### Idea: combine (merchant name, franchaise_ID, promotion, valid date, promo code -> unique ID)?
    df_final['deal_id'] = ['_'.join([country,bank_name,str(timestamp),str(i)]) for i in range(df_final.shape[0])]
    
    df_final.reset_index(inplace = True, drop = True)
    return df_final