import re
import requests
import numpy as np
from datetime import datetime

""" Remove all html tag """
def remove_html_tags(text):
    """Remove html tags from a string
    :param text: input text
    :return clean text
    """
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

""" Remove all html tag """
def remove_special_characters(text):
    """Remove special characters
    :param text: input text
    :return clean text
    """
    replace_list_wo_sp = ['<p>', '</p>','<span style="background: white; color: black;">',
                   '<p style="margin: 0in 0in 0pt;">', '<p style="margin: 0in 0in 0pt;">',
                   '</span>','<span>', '<span style="color: black;">', '</li><div>', '</div>',
                    '&ldquo;','&rdquo;','<ul style="list-style-type: circle;">', '</ul>', '<div>', 
                    '\r''<p style="margin: 0in 0in 8pt;">', '&amp;', '&reg;','^', '*' , ' *','&rsquo;s']
    for replace_word in replace_list_wo_sp:
        text = str(text).replace(replace_word, ' ')
    
    replace_list_w_sp = ['<br>', '&nbsp;', '*#']
    for replace_word in replace_list_w_sp:
        text = str(text).replace(replace_word, ' ')
        
    text = str(text).replace('&lsaquo;', '<')
    text = str(text).replace('&rsaquo;', '>')
    text = str(text).replace('&percnt;', '%')
    text = str(text).replace('&lt;', '<')
    text = str(text).replace('&gt;', '>')
    text = str(text).replace('S&dollar;', 'S$')
    text = str(text).replace('SGD', 'S$')
    return text.strip()

# """ Get sentence from a text """
# def GetSentence(text):
#     """Get sentence from text
#     :param text: input text
#     :return: sentences
#     """
#     sentences=[]
#     if len(re.findall("\n",text))>1:# & (method & 1) == 1:
#         para = re.split(r'\n', text.strip())
#         for line in para:
#             sublines = re.split(r' *[\.\?!][\'"\)\]]* *', line.strip())
#             for subline in sublines:
#                 if len(subline)>0:
#                     sentences += [subline] 
#     return sentences

# """ Get promotion code from the text """
# def GetPromoCode(text, patternlist):
#     """Get promotion code form text
#     :param text: input text
#     :param patternlist: pattern of promotion code
#     :return: promotion code
#     """
#     patterns = "|".join(patternlist)
#     srcresult = [ i for i in text.split("\n") if not re.findall(patterns,i.upper())==[]]
#     pattern=[re.findall(patterns,k.upper()) for k in srcresult]
#     pattern = list(set([y for x in pattern for y in x]))
#     if len(pattern)>0:
#         pattern = pattern[0]
#         srcresult = [(s.upper().split(pattern)[-1]).strip().split()[0] for s in srcresult]
#         for puncs in [":",";",'"',"'",",",")","]","}","(","[","{"]:
#             srcresult = [i.replace(puncs, '') for i in srcresult if len(i)>2]
#     return ' '.join(srcresult)

""" Get Issuer Exclusivity """
def get_issuer_exclusivity(terms):
    """Get issuer exclusivity from terms
    :param terms: text
    :return: issuer
    """
    dictkeys = {'visa':'visa', 'master':'mastercard|master card', 'amex':'americanexpress|american express', 'unionpay':'unionpay|union pay'}
    issuer_count = {'visa':0, 'master':0,'amex':0,'unionpay':0}
    for key,value in dictkeys.items():
        issuer_count[key] = len(re.findall(value, terms.lower()))
    if max(issuer_count.values())==0:
        return 'all'
    else:
        return  max(issuer_count, key=issuer_count.get)

def get_image(img_url, img_set, dir_path):
    """Get image from url and save to a folder
    :param img_url: url
    :param img_set: set of images
    :param dir_path: path to folder
    :return: part to image
    """
    if img_url != 'https://www.ocbc.com/assets/images/Cards_Promotions_Visuals/':
        img_name = img_url.split('/')[-1]
        if not re.search(r'\.(jpg|jpeg|png)$', img_name):
            img_name = img_name.split("\\")[0]
            if not re.search(r'\.(jpg|jpeg|png)$', img_name):
                img_name = img_name.split('?')[0]
        img_path = dir_path+ img_name
        if img_name not in img_set:
#             img_res = requests.get(img_url)
#             with open(img_path, 'wb') as handle:
#                 handle.write(img_res.content)
            
            img_set.add(img_name)    
        return img_path
    else:
        return None
    
""" Get std date time format """
def GetStdDateTime(time):
    """Get std date time format fromraw data
    :param time: input date time of raw data
    :param bank_name: name of the bank, since each bank has a different format
    :return: std_date format
    """
    try:
        time = datetime.strptime(str(time),'%Y-%m-%d %H:%M:%S').date()
    except:
        try:
            time = datetime.strptime(str(time),'%m/%d/%Y').date()
        except:
            try:
                time = datetime.strptime(str(time),'%d-%B-%y').date()
            except:
                try:
                    time = datetime.strptime(str(time),'%d-%b-%y').date()
                except:
                    try:
                        time = datetime.strptime(str(time),'%d/%m/%y').date()
                    except:
                        try:
                            time = datetime.strptime(str(time),'%d/%m/%y').date()
                        except:
                            try:
                                time = datetime.strptime(str(time).replace(',',''), '%B %d %Y').date()
                            except:
                                try:
                                    time = datetime.strptime(str(time).replace(',',''), '%d %b %Y').date()
                                except:
                                    try:
                                        time = datetime.strptime(str(time).replace(',',''), '%d %b %y').date()
                                    except:
                                        try:
                                            time = datetime.strptime(str(time).replace(',',''), '%d %B %Y').date()
                                        except:
                                            time = ''  
    return str(time)

""" Make promotion_analytic column """
def promo_caption_analysis(promotion_caption):
    promo_caption_analysis = str(promotion_caption).replace(r"1 for 1", "1-for-1")
    promo_caption_analysis = promo_caption_analysis.replace(r"1-For-1", "1-for-1")
    promo_caption_analysis = promo_caption_analysis.replace(r"Buy 1 get 1 free", "1-for-1")
                                        
    # Deal with SGD
    match = re.search('(\d+)SGD', promo_caption_analysis)
    if match:
        promo_caption_analysis = re.sub(str(match.group(1)) + "SGD", "S$" + str(match.group(1)), promo_caption_analysis)  
    promo_caption_analysis = promo_caption_analysis.replace("SGD", "S$")
    
    # Deal with USD to SGD
#     USD_to_SGD = 1.37
#     match = re.search('USD(\d+)|USD(\d+) ', promo_caption_analysis)
#     if match:
#         promo_caption_analysis = re.sub(r"USD " + str(match.group(1)), "S$" + str(int(int(match.group(1)) * USD_to_SGD)), promo_caption_analysis)
#         promo_caption_analysis = re.sub(r"USD"  + str(match.group(1)), "S$" + str(int(int(match.group(1)) * USD_to_SGD)), promo_caption_analysis)
    
#     # Deal with AED to SGD
#     AED_to_SGD = 0.31
#     match = re.search('AED(\d+)|AED(\d+) ', promo_caption_analysis)
#     if match:
#         promo_caption_analysis = re.sub(r"AED " + str(match.group(1)), "S$" + str(int(match.group(1)) * AED_to_SGD), promo_caption_analysis)
#         promo_caption_analysis = re.sub(r"AED"  + str(match.group(1)), "S$" + str(int(match.group(1)) * AED_to_SGD), promo_caption_analysis)
    
#     # Deal with HKD to SGD
#     HKD_to_SGD = 0.18
#     match = re.search('HKD(\d+)|HKD(\d+) ', promo_caption_analysis)
#     if match:
#         promo_caption_analysis = re.sub(r"HKD " + str(match.group(1)), "S$" + str(int(match.group(1)) * HKD_to_SGD), promo_caption_analysis)
#         promo_caption_analysis = re.sub(r"HKD"  + str(match.group(1)), "S$" + str(int(match.group(1)) * HKD_to_SGD), promo_caption_analysis)
    
    # Deal with JPY to SGD
    # Note: some special: JPY X|Y? What does this mean? XY?
    # e.g., ocbc_v4.csv 44 Valid from now till 31 October 2019 Enjoy 5% off + 8% Tax Free with minimum purchase of JPY 5|264 (tax exclusive)
    # Note: Valid for OCBC Visa Cardholders. Shop now:https://www.biccamera.co.jp.e.lj.hp.transer.com/bicgroup/index.html
#     JPY_to_SGD = 0.013
#     match = re.search('JPY(\d+)|JPY(\d+) | JPY (\d+)', promo_caption_analysis)
#     if match:
#         print(match)
#         promo_caption_analysis = re.sub(r"JPY " + str(match.group(1)), "S$" + str(int(match.group(1)) * JPY_to_SGD), promo_caption_analysis)
#         promo_caption_analysis = re.sub(r"JPY"  + str(match.group(1)), "S$" + str(int(match.group(1)) * JPY_to_SGD), promo_caption_analysis)
#     promo_caption_analysis.replace("1,000 JPY", "S$" + str(int(1000 * JPY_to_SGD)))
    
    # Deal with dines
    ## pattern 1: 1 dines free with every x paying adults
    match_1 = re.search('(\d+) dines free with', promo_caption_analysis)
    match_2 = re.search('(\d+) paying guests', promo_caption_analysis) 
    if match_1 and match_2:
        promo_caption_analysis = re.sub(str(match_1.group(1)) +  ' dines free with ' + str(match_2.group(1)) 
                                        + ' paying guests', " pay " + str(match_2.group(1)) + " dine " 
                                        + str(int(match_1.group(1)) + int(match_2.group(1))), promo_caption_analysis)
    
    ## patern 2:  1 dines free with x paying guests
    match_1 = re.search('(\d+) dines free with ', promo_caption_analysis)
    match_2 = re.search('(\d+) paying adults ', promo_caption_analysis)
 
    if match_1 and match_2:  
        promo_caption_analysis = re.sub(str(match_1.group(1)) + ' dines free with every ' + str(match_2.group(1))
                                      + ' paying adults', " pay " + str(match_2.group(1)) + " dine " 
                                      + str(int(match_1.group(1)) + int(match_2.group(1))), promo_caption_analysis)
    
    # Deal with with no min. spend / min. to minimum / regular-priced to regular price/ minimum
    promo_caption_analysis = promo_caption_analysis.replace("with no min. spend", "")
    promo_caption_analysis = promo_caption_analysis.replace("min.", "minimum")
    promo_caption_analysis = promo_caption_analysis.replace("with min $", "with minimum S$")
    promo_caption_analysis = promo_caption_analysis.replace("with a min spend of ", "with a minimum spend of ")
    promo_caption_analysis = promo_caption_analysis.replace("regular-priced", "regular price")
    promo_caption_analysis = promo_caption_analysis.replace("late check-...", "late check-out")
    promo_caption_analysis = promo_caption_analysis.replace("...", "")
   
    # Deal with every spend of S$X to every S$X spend
    match = re.search('every spend of S\$(\d+)', promo_caption_analysis)
    if match:
        promo_caption_analysis = re.sub(r' every spend of S\$' + str(match.group(1)), ' every S\$' + str(match.group(1)) + ' spend', promo_caption_analysis)
    # Deal with above S$X to mimimum S$X
    match = re.search('above S\$(\d+)', promo_caption_analysis)
    if match:
        promo_caption_analysis = re.sub(r'above S\$' + str(match.group(1)), 'minimum S\$' + str(match.group(1)), promo_caption_analysis)
#     # Deal with with min $500 spend
#     match = re.search('with min \$(\d+) spend', promo_caption_analysis)
#     if match:
#         promo_caption_analysis = re.sub(r'with min \$' + str(match.group(1)) + ' spend', 'minimum S\$' + str(match.group(1)), promo_caption_analysis)
        
    return promo_caption_analysis