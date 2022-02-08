# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:23:55 2022

@author: Preshita
"""

from selenium import webdriver
import time
import pandas as pd
import sqlite3 as sql
import numpy as np

url = "https://www.etsy.com/in-en/listing/535512763/sterling-silver-leaf-branch-ring-gold?click_key=941b5d563d9e218df9c8f305ebad0d7fe92f6b06%3A535512763&click_sum=0fc42dd2&ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=&ref=sr_gallery-1-5&bes=1"
#url = "https://www.etsy.com/in-en/listing/256938220/girls-t-shirt-from-friends-l-womens?click_key=f820d51f28a01496566718af73861b88acfe289d%3A256938220&click_sum=88af6c55&ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=&ref=sr_gallery-1-6&bes=1&col=1&sts=1"
#url = "https://www.etsy.com/in-en/listing/121169133/eco-friendly-felted-wool-boots-brown?ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=Minimalist&ref=sc_gallery-1-2&frs=1&listing_id=121169133&listing_slug=eco-friendly-felted-wool-boots-brown&plkey=5bb4bd9e39ac40a328f24b6dd92bb03d5d3a01f5%3A121169133"
driver = webdriver.Chrome()
driver.get(url)
review_list = []
rating=[]

org = driver.find_element_by_css_selector('#reviews > div.wt-flex-xl-5.wt-flex-wrap > nav > ul > li:nth-child(5) > a') 
val = org.get_attribute("data-page")

j=1
while j<235:
#while j<170:
#while j<70:
    time.sleep(3)
    for i in range(3):
        review1 = driver.find_elements_by_css_selector("#review-preview-toggle-"+str(i))  
        for r in review1:
            review_list.append(r.text)
            if(r.text!=""):
                rating1 = driver.find_element_by_css_selector("#same-listing-reviews-panel > div > div:nth-child("+str(i+1)+") > div.wt-pl-xs-8 > div.wt-mb-xs-1.wt-mb-md-1.wt-display-flex-md > div > div.wt-mb-xs-1 > span > span.wt-screen-reader-only")
                rating.append(rating1.get_attribute('innerHTML'))

    if driver.find_elements_by_css_selector("#review-preview-toggle-3"):
        review1 = driver.find_elements_by_css_selector("#review-preview-toggle-3")
        for r in review1:
            review_list.append(r.text)
            if(r.text!=""):
                rating1 = driver.find_element_by_css_selector("#same-listing-reviews-panel > div > div:nth-child(4) > div.wt-pl-xs-8 > div.wt-mb-xs-1.wt-mb-md-1.wt-display-flex-md > div > div.wt-mb-xs-1 > span > span.wt-screen-reader-only")
                rating.append(rating1.get_attribute('innerHTML'))

    else:
        pass
    time.sleep(3)   
    print(j)
    
    
    if j<=2 or j==int(val) or j==int(val)-1:
        nextpage = driver.find_element_by_css_selector('#reviews > div.wt-flex-xl-5.wt-flex-wrap > nav > ul > li:nth-child(6) > a').click()
    elif (j>2 and j<int(val)-1):
        nextpage = driver.find_element_by_css_selector('#reviews > div.wt-flex-xl-5.wt-flex-wrap > nav > ul > li:nth-child(7) > a').click()                                              
    j=j+1

driver.close()

#print(len(review_list))
#print(len(rating))
#print(rating)
#print(review_list)

rating_list = [w[0] for w in rating]
a = np.array(list(zip(review_list,rating_list)))
b = np.row_stack((['Review','Rating'],a))

import csv
with open("C:/Users/Preshita/Documents/FinalYearProject/DataProcessing/jewlery_final.csv", "w",newline="", encoding="utf-8" ) as f:
    writer = csv.writer(f)
    writer.writerows(b)

#conn = sql.connect('etsy_review.db')
#df.to_sql('shoes_tb1',conn)
#new_df = pd.read_sql('SELECT * FROM shoes_tb1',conn)
    
