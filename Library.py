#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:23:29 2017

@author: Beth
"""

import pandas as pd
import numpy as np

def Explore():

#    Books = pd.read_csv("/Users/Beth/Downloads/Checkouts_by_Title (7).csv")
#    Books = Books.loc[Books["Title"].apply(
#            lambda x: "Uncataloged" not in x)]
#    Books = Books.loc[Books["Title"].apply(
#            lambda x: "[videorecording]" not in x)]
#    Books = Books.loc[Books["Title"].apply(
#            lambda x: "<Unknown Title>" not in x)]
#
    #Books.to_pickle("/Users/Beth/Python/EBooks_all")
    Books = pd.read_pickle("/Users/Beth/Python/EBooks_all")
    #Books2016 = pd.read_pickle("/Users/Beth/Python/Books2016")
    
    ### Make a date field (with a datetime), from the year and month cols
    Books["Date"] = pd.to_datetime(pd.DataFrame({"Year": Books["CheckoutYear"]\
        , "Month": Books["CheckoutMonth"], "Day": [1]*len(Books)}))
    
    ### Analyze how consistent the popularity is within ebooks by the same author
    B = Books.groupby(["Creator"])["Creator","Title", "Date", "Checkouts"]
    print("Total authors: ", len(B.groups))
    interestingauthors = [] # Authors who aren't consistent!
    authorswithtwoplus = 0 # Population of authors with more than one book
    for author in [g for g in B.groups][:]:
        data = B.get_group(author).groupby(["Title"])
        ldata = len(data.sum())
        if(ldata >=4):
            authorswithtwoplus += 1
            means = []
            for title in data.groups:
                ### get only the first 6 months that we had checkouts
                tdata = data.get_group(title).sort_values(by="Date")[:6]
                tm = np.mean(tdata)
                means.append(tm)
            m = np.mean(means)
            s = np.std(means)
            if((s**2) > 5*m): 
                print(author, m, s, s**2)
                #print(means)  
                interestingauthors.append(author)
              
                
    print("Authors with two ebooks: ", authorswithtwoplus)            
    print("Authors with atypical results: ", len(interestingauthors))
    print(interestingauthors)

 
### Analyzing popularity of books using "biggest box" metric (below)
    
#    B = Books.groupby(["Title"])["Title", "Checkouts"].sum()
#    print(B.describe())
#
#    Bpop = B.loc[lambda x: x["Checkouts"]>500]
#    print(Bpop)
#
#    bestMonth, biggestBox = BiggestBox(Bpop.index, Books)
#    
#    bestMonth = bestMonth.sort_values(by="BestMonth", ascending=False)
#    print(bestMonth[:20])
#    
#    biggestBox = biggestBox.sort_values(by="BiggestBox", ascending=False)
#    print(biggestBox[:20])
    
### Trying to split kids books from adult books, based on "Subject" column
    
#    KidsBooks = Books.loc[Books["Subjects"].apply(
#            lambda x: "Juvenile" in str(x))]
#    #print(KidsBooks.count())
#    AdultBooks = Books.loc[Books["Subjects"].apply(
#            lambda x: "Juvenile" not in str(x) and
#            "Stories in rhyme" not in str(x))]
#    #print(AdultBooks.count())
#    
#    A = AdultBooks.groupby(["Title"])["Title","Checkouts"].sum()
#    print("Adult Books:", A.describe())
#    Apop = A.loc[lambda x: x["Checkouts"] > 500]
#    print(Apop)


### A method to reduce popularity stats to only two values!
###     This assumed that all the results were from the same year
def BiggestBox(titles, Books):
    
    bestMonth = pd.DataFrame(columns = ["Title","BestMonth"])
    biggestBox = pd.DataFrame(columns = ["Title", "BiggestBox"])
    rnum = 0
    for title in titles:
        
        ### Grouping by month lets us combine records for regular and 
        ###     large print editions, etc.
        data =  Books.loc[Books["Title"].apply(lambda x: title in str(x))]\
                    .groupby(["CheckoutMonth"])["Checkouts"].sum()
        data = data.sort_values()
        L = len(data)
        #print(data)
        maxval = 0
        pos = 0
        for i in range(len(data)):
            #print(L-i, data.iloc[i])
            potential = data.iloc[i]*(L-i)
            if(potential > maxval):
                maxval = potential
                pos = L-i
        print(maxval, pos, data.iloc[L-pos], title)
        rec = pd.DataFrame({"Title": title, "BestMonth": data.iloc[L-pos]}, 
                            index = [rnum])
        #print(rec)
        bestMonth = bestMonth.append(rec)
        #print(bestMonth)
        rec2 = pd.DataFrame({"Title": title, "BiggestBox": maxval}, 
                            index = [rnum])
        biggestBox = biggestBox.append(rec2)

        rnum += 1
        
    print(bestMonth)
    print(biggestBox)
    return(bestMonth, biggestBox)
    
Explore()
