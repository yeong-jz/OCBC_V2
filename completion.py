import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
    
""" Remove std category """
def get_stdcat_from_tax(terms, taxonomy):
    count = []
    for key in list(taxonomy.keys()):
        count_vect = CountVectorizer(lowercase = True, vocabulary = taxonomy[key] )
        a = count_vect.fit_transform([terms])
        count += [np.sum(a)]
    return list(taxonomy.keys())[np.argmax(count)]

""" Completion pipeline: CC cat"""
def completion_CCcat(df_card_name, df_cat, df_subcat, df_merchant, df_promo, df_term, cat_to_CCcat, CC_category_taxonomy):
    if re.findall('_', df_card_name)!=[]:
        bank_name = df_card_name.split('_')[0]
    else:
        bank_name = 'dbs'
    
    # separate case for citi and scb because we also look at subcategory
    if bank_name in ['citi', 'scb']:
        cat_key = str(df_cat).lower()+'#'+str(df_subcat).lower()
    else:
        cat_key = str(df_cat).lower()
    if cat_to_CCcat[bank_name][cat_key]['apply_tax']=='True':
        df_cat = get_stdcat_from_tax(' '.join([str(df_merchant), str(df_promo),str(df_term)]), CC_category_taxonomy)
    else :
        df_cat = cat_to_CCcat[bank_name][cat_key]['std_category']
    
    return df_cat
   
""" Completion pipeline: stdcat"""
def completion_stdcat(df_card_name, df_cat, df_subcat, df_merchant, df_promo, df_term, cat_to_stdcat, std_category_taxonomy):
    if re.findall('_', df_card_name)!=[]:
        bank_name = df_card_name.split('_')[0]
    else:
        bank_name = 'dbs'
    # separate case for citi and scb because we also look at subcategory
    if bank_name in ['citi', 'scb']:
        cat_key = str(df_cat).lower()+'#'+str(df_subcat).lower()
    else:
        cat_key = str(df_cat).lower()
    if cat_to_stdcat[bank_name][cat_key]['apply_tax']=='True':
        df_cat = get_stdcat_from_tax(' '.join([str(df_merchant), str(df_promo),str(df_term)]), std_category_taxonomy)
    else :
        df_cat = cat_to_stdcat[bank_name][cat_key]['std_category']
    
    return (bank_name, df_cat)
    
""" Completion pipeline: google type
    Do after the completion pipeline of standard_category or CC Buddy Category
"""
def completion_google_type(standard_category, stdcategory_to_googletype):
    cat_key = str(standard_category).lower()
    df_cat = stdcategory_to_googletype[cat_key]    
    return df_cat

""" Completion pipeline: google_api """
def completion_google_api(df_address, df_is_online):
    if str(df_address) == "":
        if str(df_is_online) == "True":
            return (False, None)
        else:
            if str(df_is_online) == "False":
                return (True, None)
            else:
                return (None, None)
    else:
        match = re.search(r'www|website|website.|participating outlets|http|https|Valid at all|Click here|View here', str(df_address))
        if match:
            return (True, str(df_address).lower())
        else:
            return (False, None)    
    
""" Completion pipeline: postal code """
def completion_postal(is_online, postal_code, postal_code_map):
    if str(postal_code).isdigit():
        num_list = list(str(postal_code).zfill(6))
        sector_num = int("".join(num_list[0:2])) # two-first numbers -> sector
        sector = sector_num
        if sector_num in list(postal_code_map.postal_sector):   
            district = postal_code_map[postal_code_map.postal_sector == sector_num]['postal_district'].iloc[0]
            district_name = postal_code_map[postal_code_map.postal_sector == sector_num]['postal_district_name'].iloc[0]
            general_location = postal_code_map[postal_code_map.postal_sector == sector_num]['general_location'].iloc[0]
            area = postal_code_map[postal_code_map.postal_sector == sector_num]['suggested_area'].iloc[0]
        else:
            district, district_name, general_location, area = '', '', '', ''
    else:
        if str(is_online) == "True":
            postal_code, sector, district, district_name, general_location, area = '', '', '', 'Online', 'Online', 'Online' 
        else:
            postal_code, sector, district, district_name, general_location, area = '', '', '', '', '', ''
            
    if str(sector).isdigit():
        return str(postal_code).zfill(6), str(sector).zfill(2) , str(district).zfill(2), str(district_name), str(general_location), str(area)
    else:
        return str(postal_code), str(sector), str(district), str(district_name), str(general_location), str(area)