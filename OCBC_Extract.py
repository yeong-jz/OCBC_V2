## Load packages
import re
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from os import listdir, makedirs
from os.path import isfile, join, exists
from  bs4    import BeautifulSoup
from  urllib.request import Request,urlopen
from helper_func import *
from completion import *
from google_api_caller import *


import time
start_time = time.time()

import boto3
import s3fs

s3 = s3fs.S3FileSystem(anon=False)

""" Strip all html tag """
def stripHtmlTags(htmlTxt):
    """Strip html tags from a string
    :param text: input text
    :return clean text
    """
    if htmlTxt is None:
            return None
    else:
        return ' '.join(BeautifulSoup(htmlTxt, features="lxml").findAll(text=True))
    
""" Get is online or not """
def isOnline(cat, subcat):
    if re.search(r'online', str(cat)) or re.search(r'online', str(subcat)):
        return True
    else:
        return False

def GetPromoCode(text, patternlist):
    """Get promotion code form text
    :param text: input text
    :param patternlist: pattern of promotion code
    :return: promotion code
    """
    patterns = "|".join(patternlist)
    srcresult = [ i for i in text.split("\n") if not re.findall(patterns,i.upper())==[]]
    pattern=[re.findall(patterns,k.upper()) for k in srcresult]
    pattern = list(set([y for x in pattern for y in x]))
    if len(pattern)>0:
        pattern = pattern[0]
        srcresult = [(s.upper().split(pattern)[-1]).strip().split()[0] for s in srcresult]
        for puncs in [":",";",'"',"'",",",")","]","}","(","[","{"]:
            srcresult = [i.replace(puncs, '') for i in srcresult if len(i)>2]
    return ' '.join(srcresult)

def GetValidity(start, end, terms):
    """ In the case when we don't have the startGet std date time format from raw data
    :param time: input date time of raw data
    :param bank_name: name of the bank, since each bank has a different format
    :return: std_date format
    """
    validity = str(terms)
    # Get the all search values
    validity_all = re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{4}", validity)
    # Get the end from search values
    if not re.search(r"booking period|stay period", validity):
        if len(validity_all) == 1:
            validity_start = ""
            validity_end = validity_all[0]
        elif len(validity_all) == 2:
            validity_start = validity_all[0]
            validity_end = validity_all[1]
        else:
            validity_start = start
            validity_end   = end
    else:
        validity_start = start
        validity_end = end
    # Standalization
    str_validity_start = GetStdDateTime(validity_start)
    std_validity_end = GetStdDateTime(validity_end)
    return np.array([str_validity_start, std_validity_end, terms])

def GetPostalCode(address):
    """Get postal code from address
    :param address: the address of merchant
    :return: postal code
    """
    IsSingapore = True if len(re.findall('Singapore', str(address))) else False
    pc = ""
    if IsSingapore:
        pc = str(address.split("Singapore")[-1].strip())
    return pc

def GetPhoneNumber(phone):
    """Get the phone number (format: xxxx xxx)
    :param phone: phone or text in str
    :return: phone number
    """
    pattern= r"\D(\d{4})\D(\d{4})\D"
    phone = re.findall(pattern,str(phone))
    return ('/'.join(phone))

""" Get min pax and max pax"""
def GetMaxPax(terms):           
    try:
        max_pax = re.findall(r'[mM]ax[a-zA-Z ]* \d{1,2} p[a-zA-Z]*', terms)[0]
        max_pax = re.findall(r'\d{1,2}', max_pax)[0]
    except:
        max_pax = np.nan
    return (max_pax)

def GetMinPax(terms):           
    try:
        min_pax = re.findall(r'[mM]in[a-zA-Z ]* \d{1,2} p[a-zA-Z]*', terms)[0]
        min_pax = re.findall(r'\d{1,2}', min_pax)[0]
    except:
        min_pax = np.nan
    return (min_pax)

def address_process(address):
    match = re.findall(r"\d{6}", str(address))
    if len(match) >1:
        for j in range(len(match)):
            address = address.replace(str(match[j]), str(match[j]) + " * ")
    return address

def GetAddress(address):
    match = re.findall(r"\d{6}", str(address))
    if len(match) >1:
        for j in range(len(match)):
            address = address.replace(str(match[j]), str(match[j]) + " * ")
    return address

def ExactPage(deal_url):
    """Exact information from the page
    :param deal_url: url of deal
    :return: data frame of deals after collecting data
    """
    try:        
        req = Request(deal_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        dealtxt = stripHtmlTags(webpage)
        p   = re.compile(r'^\d+,', flags=re.MULTILINE)
        pos, c, ps, pa  = 0, 0, 1, [] 
        while True:
            match = p.search(dealtxt, pos)
            if not match:
                break
            s = match.start()
            e = match.end()
            pos = e
            c   += 1
            if c > 0:
                line = dealtxt[ps:s]
                pa.append(line.split(","))
            ps  = s
        pa   = np.array(pa)
        return pa
    except Exception as e:
        print(deal_url,": Not Found")
        print(e)
        return ""
    
""" Get the deals """
def GetDeals(url):
    """Get the deals from url and do preprocessing data
    :param url: url of deal
    :param Deals: data frame of deals (= None)
    :return: data frame of deals after collecting data
    """
    pa   = ExactPage(url)
    df_deals = pd.DataFrame(data=pa[1:, 0:], columns=pa[0, 0:])
    OCBCDeals = pd.DataFrame()
    OCBCDeals = pd.concat([df_deals["Merchant"], df_deals["main"], df_deals["Subcategory"],
                          df_deals["Long Promo"], df_deals["Short promo"], df_deals["Website"],
                          df_deals["Address"],df_deals["Telephone"], df_deals["Start Date"],
                          df_deals["End Date"], df_deals["Terms"], df_deals["Small Image"],
                          ],  axis=1)
    OCBCDeals.columns = ['merchant_name','category','subcategory', 'promotion','promotion_caption',
                         'website', 'address','phone','start', 'end','terms', 'image_url']
    
    OCBCDeals['image_url']           = df_deals[["Small Image", "Hero Image"]].apply(lambda x: x[0] if x[0] != '' else x[1], axis=1)
    
    OCBCDeals.address                = OCBCDeals.address.apply(lambda x: address_process(x))
    OCBCDeals                        = OCBCDeals.assign(address = OCBCDeals.address.apply(lambda x: str(x).split(" *")[:-1] if len(str(x).split(" *")) >1 else x)).explode('address').reset_index(drop=True)
    
    OCBCDeals['start']               = OCBCDeals[['start', 'end', 'terms']].apply(lambda x: GetValidity(x.start, x.end, x.terms)[0], axis=1)
    OCBCDeals['end']                 = OCBCDeals[['start', 'end', 'terms']].apply(lambda x: GetValidity(x.start, x.end, x.terms)[1], axis=1)
    OCBCDeals['merchant_compressed'] = OCBCDeals.merchant_name.apply(lambda x: str(x).lower().replace(' ',''))
    OCBCDeals['postal_code']         = OCBCDeals.address.apply(lambda x: GetPostalCode(x))
    patternlist                      = ['PROMO CODE:','PROMO CODE :','PROMO CODE']
    OCBCDeals['promo_code']          = OCBCDeals.promotion.apply(lambda x: GetPromoCode(x,patternlist))
    OCBCDeals['flag']                = OCBCDeals.promotion.apply(lambda x: "")
    OCBCDeals['comments']                = OCBCDeals.promotion.apply(lambda x: "")
    OCBCDeals['issuer_exclusivity']  = OCBCDeals.terms.apply(lambda x: get_issuer_exclusivity(x))
    OCBCDeals['card_name']           = OCBCDeals.terms.apply(lambda x: 'ocbc_' + get_issuer_exclusivity(x))
    OCBCDeals['latitude']            = OCBCDeals.terms.apply(lambda x: None)
    OCBCDeals['longitude']           = OCBCDeals.terms.apply(lambda x: None)
    OCBCDeals['max_pax']             = OCBCDeals.terms.apply(lambda x: GetMaxPax(x))
    OCBCDeals['min_pax']             = OCBCDeals.terms.apply(lambda x: GetMinPax(x))
    OCBCDeals['is_online']           = OCBCDeals[['category','subcategory']].apply(lambda x: isOnline(x.category, x.subcategory), axis=1)
    OCBCDeals['category']            = OCBCDeals['category'].apply(lambda x: x.split('|')[0] if x.split('|')[0] != '' else np.nan)
    
    OCBCDeals['raw_input']           = OCBCDeals.category.apply(lambda x: None)
    OCBCDeals['promotion_analytic']  = OCBCDeals.promotion_caption.apply(lambda x: promo_caption_analysis(x))
    img_url ='https://www.ocbc.com/assets/images/Cards_Promotions_Visuals/'
    OCBCDeals['image_url']           = OCBCDeals.image_url.apply(lambda x: img_url + str(x))
    
    ### Image directory ###
    img_dir='images/ocbc/' + str(date.today()) + '/'
    if not exists(img_dir):
        makedirs(img_dir)
    img_set = set([f for f in listdir(img_dir) if isfile(join(img_dir, f))])
    OCBCDeals["image_path"]=OCBCDeals.image_url.apply(lambda x:get_image(x,img_set,img_dir)) 
    
    ### Standarlization ###
    OCBCDeals['google_api']     = OCBCDeals[['address', 'is_online']].apply(lambda x: completion_google_api(x.address, x.is_online)[0], axis=1)
    OCBCDeals['listing_outlet'] = OCBCDeals[['address', 'is_online']].apply(lambda x: completion_google_api(x.address, x.is_online)[1], axis=1)
    
    OCBCDeals['std_category']   = OCBCDeals[['card_name', 'category', 'subcategory', 'merchant_name', 'promotion', 'terms']].apply(lambda x: 
            completion_stdcat(str(x.card_name), str(x.category), str(x.subcategory), str(x.merchant_name), 
                              str(x.promotion), str(x.terms), cat_to_stdcat, std_category_taxonomy)[1], axis=1)
    OCBCDeals['cc_buddy_category'] = OCBCDeals[['card_name', 'category', 'subcategory', 'merchant_name', 'promotion', 'terms']].apply(lambda x: 
            completion_CCcat(str(x.card_name), str(x.category), str(x.subcategory), str(x.merchant_name), 
                             str(x.promotion), str(x.terms), cat_to_CCcat, CC_category_taxonomy), axis=1)   
    OCBCDeals['google_type'] = OCBCDeals.std_category.apply(lambda x: completion_google_type(x, stdcategory_to_googletype)) 
   
    ### Google API ###
    
    ## Postal Code ###
    
    OCBCDeals['postal_code']      = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[0]), axis=1)
    OCBCDeals['country']          = OCBCDeals['postal_code'].apply(lambda x: "SGP")
    OCBCDeals['sector']           = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[1]), axis=1)
    OCBCDeals['district']         = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[2]), axis=1)
    OCBCDeals['district_name']    = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[3]), axis=1)
    OCBCDeals['general_location'] = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[4]), axis=1)
    OCBCDeals['area']             = OCBCDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[5]), axis=1)
    
    OCBCDeals.reset_index(inplace = True, drop = True)
    return OCBCDeals

'''                                         Functions for Changi                                       '''

def get_url_content(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    return webpage

def get_p_merchants(url):
    webpage = get_url_content(url)
    souptxt = BeautifulSoup(webpage, 'html.parser')
    merchants    = souptxt.findAll("a", attrs={"class": "tingle ajax"})
    rows=[]
    count=0
    for mer in merchants:
        tc_url = mer.get("data-url")
        tc = get_url_content(tc_url.split("#")[0])
        sptxt = BeautifulSoup(tc, 'html.parser')
        ps  = sptxt.findAll("div", attrs={"class": "cmp-text"})
        tc=[]
        #Dylan
        dates =[]
        for t in ps:
            tc.append(re.sub('\s+', ' ', t.getText()).strip())
            dates.append(t.find("li").getText())    
        ps = mer.findAll("p")
        terminal=ps[0].getText()
        name = mer.find("h3").getText()
        desc = ps[1].getText()
        terminal=re.sub('\s+', ' ', terminal).strip()
        name=re.sub('\s+', ' ', name).strip()
        desc = re.sub('\s+', ' ', desc).strip()
        image = "http://www.changiairport.com" + mer.find("figure").find("img").get("src")
        row=[name,terminal,desc,"\n - ".join(tc),image, dates[0]] # add dates[0] if we want to get raw dates
        rows.append(row)
        count+=1
    p_merchant = pd.DataFrame(rows,columns=('Merchant', 'Terminal', 'promotion','terms',"image","dates")) # <--"dates" if we want dates
    return p_merchant

def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, list):
              t.append(i)
        else:
              t.extend(flatten(i))
    return t

def start_date_config(dates):
    #remove all the 'only' in the date string
    dates = dates.replace('only', '')
    dates = dates.replace('Only', '')
    
    #split the sentence into a list
    dates= dates.split(" ")
    
    # deal with dec2019
    regex = re.compile(r'(\d+|\s+)')
    dates[-1] = regex.split(dates[-1])
    dates = flatten(dates)

    #remove all the empty''
    dates = list(filter(None, dates)) 

    #if last element is 2019Â
    if dates[-1].__contains__('Â'):
        temp = dates[-1]
        temp = temp[:-1]
        dates[-1] = temp
        st = dates[2] + " " + dates[3] + " " + dates[7]
        start_date = datetime.strptime(st, '%d %b %Y')
        start_date = start_date.strftime('%-m/%-d/%Y')
    else:
        st = dates[2] + " " + dates[3] + " " + dates[7]
        start_date = datetime.strptime(st, '%d %b %Y')
        start_date = start_date.strftime('%-m/%-d/%Y')

    return start_date
    
def end_date_config(dates):
    #remove all the 'only' in the date string
    dates = dates.replace('only', '')
    dates = dates.replace('Only', '')
    
     #split the sentence into a list
    dates= dates.split(" ")
    
    # deal with dec2019
    regex = re.compile(r'(\d+|\s+)')
    dates[-1] = regex.split(dates[-1])
    dates = flatten(dates)

    #remove all the empty''
    dates = list(filter(None, dates)) 


    #if last element has Â
    if dates[-1].__contains__('Â'):
        temp = dates[-1]
        temp = temp[:-1]
        dates[-1] = temp
        et = dates[5] + " " + dates[6] + " " + dates[7]
        end_date = datetime.strptime(et, '%d %b %Y')
        end_date = end_date.strftime('%-m/%-d/%Y')
    else:
        et = dates[5] + " " + dates[6] + " " + dates[7]
        end_date = datetime.strptime(et, '%d %b %Y')
        end_date = end_date.strftime('%-m/%-d/%Y')

    return end_date


# main
if  __name__  ==  '__main__' :   
    data_folder = "data/"
    """ Load merchant_dict """
    with open(data_folder + 'merchant_dict_clean.pickle', 'rb') as handle:
        merchant_dict = pickle.load(handle)

    """ Load cat_to_stdcat  for CC cat """
    with open(data_folder + 'cat_to_CC_cat.pickle', 'rb') as handle:
        cat_to_CCcat = pickle.load(handle)
    """ Load stdcategory_to_googletype  for CC cat """
    with open(data_folder + 'CC_category_to_googletype.pickle', 'rb') as handle:
        CCcategory_to_googletype = pickle.load(handle)
    """ Load std_category_taxonomy  for CC cat """
    with open(data_folder + 'CC_category_taxonomy.pickle', 'rb') as handle:
        CC_category_taxonomy = pickle.load(handle)

    """ Load cat_to_stdcat """
    with open(data_folder + 'cat_to_stdcat.pickle', 'rb') as handle:
        cat_to_stdcat = pickle.load(handle)
    """ Load stdcategory_to_googletype """
    with open(data_folder + 'stdcategory_to_googletype.pickle', 'rb') as handle:
        stdcategory_to_googletype = pickle.load(handle)
    """ Load std_category_taxonomy """
    with open(data_folder + 'std_category_taxonomy.pickle', 'rb') as handle:
        std_category_taxonomy = pickle.load(handle)

    postal_code_map = pd.read_csv(data_folder + 'RegionTable.csv')
    
    
    url= "https://www.ocbc.com/assets/data/card-promotions.csv?1558490527925"
    OCBCDeals = GetDeals(url)
    ColStandard = ['card_name', 'category', 'subcategory', 'cc_buddy_category', 'std_category', 'merchant_name', 
                    'merchant_compressed', 'google_type', 'promotion', 'promotion_caption','promotion_analytic', 'promo_code', 'address', 
                    'latitude', 'longitude', 'start', 'end', 'phone', 'website', 'image_url', 'image_path', 'issuer_exclusivity', 
                    'raw_input','min_pax','max_pax', 'is_online', 'listing_outlet', 'google_api', 'terms', 'postal_code', 'country',
                    'sector', 'district', 'district_name', 'general_location', 'area', 'flag', 'comments']
    StandardDeals = OCBCDeals[ColStandard]
    StandardDeals.to_csv("ocbc_" + str(date.today()) +".csv",index=False)
    
    #write to S3 ocbc file
    name = "ocbc_" + str(date.today()) +".csv"
    directory = "s3://data-pipeline-cardspal/"+str(date.today())+"/extracts/"+ name
    with s3.open(directory,'w') as f:
        StandardDeals.to_csv(f)
    print("OCBC uploaded.")

    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    
    #================                 Changi                  =================
    
    currdate=datetime.today().strftime('%Y%m%d') 
    url="http://www.changiairport.com/en/shop/promotions/mastercard-privileges.html"
    p_merchants  = get_p_merchants(url)
    p_merchants.to_csv("changi_" + currdate + ".csv")


    print("changi scrapped")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    
    ChangiDeals = p_merchants
    ChangiDeals.head()
    
    #T1,2,3,4 PUBLIC AND TRANSIT
    # These generalize addresses are just a way to link up with the merchant db to take the actual address in the merchant db

    Changi_Lat_Dict = {'T1 PUBLIC': '1.363660', 
                           'T1 TRANSIT': '1.363660', 
                           'T1 PUBLIC AND TRANSIT': '1.363660',

                           'T2 PUBLIC': '1.356909', 
                           'T2 TRANSIT': '1.356909', 
                           'T2 PUBLIC AND TRANSIT': '1.356909',

                           'T3 PUBLIC': '1.358218', 
                           'T3 TRANSIT': '1.358218', 
                           'T3 PUBLIC AND TRANSIT': '1.358218',

                           'T4 PUBLIC': '1.339913', 
                           'T4 TRANSIT': '1.339913', 
                           'T4 PUBLIC AND TRANSIT': '1.339913'
                          }
    Changi_Long_Dict = {'T1 PUBLIC': '103.990051', 
                           'T1 TRANSIT': '103.990051', 
                           'T1 PUBLIC AND TRANSIT': '103.990051',

                           'T2 PUBLIC': '103.988251', 
                           'T2 TRANSIT': '103.988251', 
                           'T2 PUBLIC AND TRANSIT': '103.988251',

                           'T3 PUBLIC': '103.986553', 
                           'T3 TRANSIT': '103.986553', 
                           'T3 PUBLIC AND TRANSIT': '103.986553',

                           'T4 PUBLIC': '103.983884', 
                           'T4 TRANSIT': '103.983884', 
                           'T4 PUBLIC AND TRANSIT': '103.983884'
                          }
    Changi_Address_Dict =  {'T1 PUBLIC': '(T1 PUBLIC) 80 Airport Boulevard, Changi Airport Terminal 1, Singapore 819642', 
                            'T1 TRANSIT': '(T1 TRANSIT) 80 Airport Boulevard, Changi Airport Terminal 1, Departure/Transit Lounge, Singapore 819642', 
                            'T1 PUBLIC AND TRANSIT': '(T1 PUBLIC AND TRANSIT) 80 Airport Boulevard, Changi Airport Terminal 1, Singapore 819642',

                            'T2 PUBLIC':'(T2 PUBLIC) 60 Airport Boulevard, Changi Airport, Terminal 2, Singapore 819643',
                            'T2 TRANSIT': '(T2 TRANSIT) 60 Airport Boulevard, Changi Airport, Terminal 2, Singapore 819643', 
                            'T2 PUBLIC AND TRANSIT': '(T2 PUBLIC AND TRANSIT) 60 Airport Boulevard, Changi Airport, Terminal 2, Singapore 819643',

                            'T3 PUBLIC': '(T3 PUBLIC) 65 Airport Blvd, Changi Airport Terminal 3, Singapore 819663', 
                            'T3 TRANSIT': '(T3 TRANSIT) 65 Airport Blvd, Changi Airport Terminal 3, Singapore 819663', 
                            'T3 PUBLIC AND TRANSIT': '(T3 PUBLIC AND TRANSIT) 65 Airport Blvd, Changi Airport Terminal 3, Singapore 819663',

                            'T4 PUBLIC': '(T4 PUBLIC) 10 Airport Blvd, Changi Airport, Terminal 4, Singapore 819665', 
                            'T4 TRANSIT': '(T4 TRANSIT) 10 Airport Blvd, Changi Airport, Terminal 4, Singapore 819665', 
                            'T4 PUBLIC AND TRANSIT': '(T4 PUBLIC AND TRANSIT) 10 Airport Blvd,Changi Airport, Terminal 4, Singapore 819665'
                           }
    
    # pre handling expansion of public, transit
    flagged = []
    for i in range(len(ChangiDeals)):
        if "AND" in ChangiDeals.Terminal.iloc[i]:
            flagged.append(i) 

    df_flagged = ChangiDeals.loc[flagged]  

    df_public = df_flagged
    df_transit = df_flagged

    df_public = df_public.reset_index(drop=True)
    df_transit = df_transit.reset_index(drop=True)
    for i in range(len(df_public)):        
            df_public.at[i, 'Terminal'] = df_public.Terminal.iloc[i][:2] + " PUBLIC"

    for i in range(len(df_transit)):        
            df_transit.at[i, 'Terminal'] = df_transit.Terminal.iloc[i][:2] + " TRANSIT"

    ChangiDeals.drop(flagged, inplace=True)

    ChangiDeals = pd.concat([ChangiDeals, df_public])
    ChangiDeals= pd.concat([ChangiDeals, df_transit])

    ChangiDeals = ChangiDeals.reset_index(drop=True)
    
    ChangiDeals['address']  = ChangiDeals.Terminal.map(Changi_Address_Dict) 
    ChangiDeals['latitude'] = ChangiDeals.Terminal.map(Changi_Lat_Dict) 
    ChangiDeals['longitude'] = ChangiDeals.Terminal.map(Changi_Long_Dict) 

    #XXXXStart, End is derived from GetValiditiy - use the existing function used in ocbc
    #Instead use new date column generated from obtaining the first find of <li> in the extract code then parse through to get start and end dates with my functions
    ChangiDeals['start'] = ChangiDeals.dates.apply(lambda x: start_date_config(x))
    ChangiDeals['end'] = ChangiDeals.dates.apply(lambda x: end_date_config(x))

    #Merchant names
    ChangiDeals['merchant_name'] = ChangiDeals['Merchant']
    ChangiDeals['merchant_compressed'] = ChangiDeals.merchant_name.apply(lambda x: str(x).lower().replace(' ',''))

    #Promocode : use OCBC function to derive PromoCode
    patternlist                      = ['PROMO CODE:','PROMO CODE :','PROMO CODE']
    ChangiDeals['promo_code']          = ChangiDeals.promotion.apply(lambda x: GetPromoCode(x,patternlist)) 
    #theres no promo code given in the promotion hence its empty

    #IssuerExclusivity 
    ChangiDeals['issuer_exclusivity']  = ChangiDeals.terms.apply(lambda x: get_issuer_exclusivity(x))

    #Min, Max : Use the already derived function in OCBC which mines the Terms and condition to get info.
    ChangiDeals['max_pax']             = ChangiDeals.terms.apply(lambda x: GetMaxPax(x))
    ChangiDeals['min_pax']             = ChangiDeals.terms.apply(lambda x: GetMinPax(x))


    #Is_online set as False
    ChangiDeals['is_online'] = False


    ChangiDeals['postal_code']         = ChangiDeals.address.apply(lambda x: GetPostalCode(x))

    ChangiDeals['card_name']           = ChangiDeals.terms.apply(lambda x: 'ocbc_' + get_issuer_exclusivity(x))

    #Promotion_analytics : use the same function as ocbc
    #There is no promotion caption in the changi data hence we just repopulate with promotion itself since its already so short anyways
    ChangiDeals['promotion_caption'] = ChangiDeals['promotion']   
    ChangiDeals['promotion_analytic']  = ChangiDeals.promotion_caption.apply(lambda x: promo_caption_analysis(x))

    ChangiDeals['phone'] = ""   
    ChangiDeals['website'] = ""      
    ChangiDeals['image_url'] = ChangiDeals['image']   

    ### Image directory ###
    img_dir='images/ocbc_changi/' + str(date.today()) + '/'
    if not exists(img_dir):
        makedirs(img_dir)
    img_set = set([f for f in listdir(img_dir) if isfile(join(img_dir, f))])
    ChangiDeals["image_path"]=ChangiDeals.image_url.apply(lambda x:get_image(x,img_set,img_dir)) 

    ### Standarlization ###
    ChangiDeals['google_api']     = ChangiDeals[['address', 'is_online']].apply(lambda x: completion_google_api(x.address, x.is_online)[0], axis=1)
    ChangiDeals['listing_outlet'] = ChangiDeals[['address', 'is_online']].apply(lambda x: completion_google_api(x.address, x.is_online)[1], axis=1)

    #Subcategory leave it as blank
    ChangiDeals['subcategory'] = np.nan
    ChangiDeals['category'] = np.nan


    ChangiDeals['std_category']   = ChangiDeals[['card_name', 'category', 'subcategory', 'merchant_name', 'promotion', 'terms']].apply(lambda x: completion_stdcat(str(x.card_name), 
                                    str(x.category), str(x.subcategory), str(x.merchant_name), str(x.promotion), str(x.terms), cat_to_stdcat, std_category_taxonomy)[1], axis=1)
    ChangiDeals['cc_buddy_category'] = ChangiDeals[['card_name', 'category', 'subcategory', 'merchant_name', 'promotion', 'terms']].apply(lambda x: completion_CCcat(str(x.card_name), 
                                    str(x.category), str(x.subcategory), str(x.merchant_name), str(x.promotion), str(x.terms), cat_to_CCcat, CC_category_taxonomy), axis=1)   
    ChangiDeals['google_type'] =  ChangiDeals.std_category.apply(lambda x: completion_google_type(x, stdcategory_to_googletype)) 

    ### Google API ###

    ## Postal Code ###

    ChangiDeals['postal_code']      = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[0]), axis=1)
    ChangiDeals['country']          = ChangiDeals['postal_code'].apply(lambda x: "SGP")
    ChangiDeals['sector']           = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[1]), axis=1)
    ChangiDeals['district']         = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[2]), axis=1)
    ChangiDeals['district_name']    = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[3]), axis=1)
    ChangiDeals['general_location'] = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[4]), axis=1)
    ChangiDeals['area']             = ChangiDeals[['is_online', 'postal_code']].apply(lambda x: str(completion_postal(x.is_online, x.postal_code, postal_code_map)[5]), axis=1)

    #Basically raw input is always empty and for flag we put as value 1 so that we can identify it when we have to (post combine with OCBC deals??)
    ChangiDeals['raw_input'] = "" 
    ChangiDeals['flag']      = "1"
    ChangiDeals['comments'] = ""

    ColStandard = ['card_name', 'category', 'subcategory', 'cc_buddy_category', 'std_category', 'merchant_name', 
                    'merchant_compressed', 'google_type', 'promotion', 'promotion_caption','promotion_analytic', 'promo_code', 'address', 
                    'latitude', 'longitude', 'start', 'end', 'phone', 'website', 'image_url', 'image_path', 'issuer_exclusivity', 
                    'raw_input','min_pax','max_pax', 'is_online', 'listing_outlet', 'google_api', 'terms', 'postal_code', 'country',
                    'sector', 'district', 'district_name', 'general_location', 'area', 'flag', 'comments']
    ChangiStandardDeals = ChangiDeals[ColStandard]
    ChangiStandardDeals.to_csv("ocbc_changi_" + str(date.today()) +".csv",index=False)
    
    #write to S3 ocbc file
    name = "changi_" + str(date.today()) +".csv"
    directory = "s3://data-pipeline-cardspal/"+str(date.today())+"/extracts/"+ name
    with s3.open(directory,'w') as f:
        ChangiStandardDeals.to_csv(f)
    print("Output Success")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    
    
