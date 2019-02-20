from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql.functions import col, size


#Creating the spark session and the SQL CONTEXT using hive
spark = SparkSession.builder.appName("infinityTweet").config("spark.jars", 'file:///media/neuronelab/6f7e1e95-2147-4fab-98df-5c23928e0fde/spark-solr/target/spark-solr-3.7.0-SNAPSHOT-shaded.jar').getOrCreate()
sc = spark.sparkContext
sqlContext = HiveContext(sc)

from tqdm import tqdm
import numpy as np
import pprint
import time
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import itertools as itertools
import re
import pandas as pd

#Function to find the next prime 
def findNextPrime(n):
    for p in range(n, n*2):
        for i in range(2, p):
            if p % i == 0:
                break
        else:
            return p
    return None

def hash(a, b,prime):
    return (ascii(a) ^ ascii(b)) % prime

def secondHash(a,  b , prime , prime2):
    return (ascii(a) ^ ascii(b) % prime2) + prime

def ascii(st):
    res = 0
    for i in st:
        res += ord(i)
    return res


def findMinSupport(findIn , t):
    toReturn = dict()
    for key , val in findIn.items():
        if val >= t:
            toReturn[key] = val
    return toReturn

treshold = 0.01

#We load from solr in the pc2 the collection infinityTweet
loadDF = sqlContext.read.format("solr").option("zkHost","172.16.175.155:9983").option("collection","infinityTweet").load()
#We select from loadDF only the text where the text of the tweet is not null ann we filter it by language ("en") using the spark context
infinityDF = sc.parallelize(loadDF.select("text").where(loadDF.text.isNotNull()).where(loadDF.lang=="en").collect())
#Delete loadDF in order for the memory
del loadDF

#we define the support as the lenght of the dataframe * threshold 
support = infinityDF.count()*treshold
#Stop words , we add some useless word
stopWords = list(stopwords.words('english'))
stopWords.append('I')
stopWords.append('-')
stopWords.append("'")
stopWords.append("rt")
stopWords.append("&amp;")
#stop words as a set
stopWordsSet = set(stopWords)
#next prime from the count of the DF
nextPrime = findNextPrime(infinityDF.count())
#nextPrime2 = findNextPrime(nextPrime)
nextPrime2 = nextPrime *2
count = infinityDF.count()
print(count)

#Creating the items dictionary and the hashmap for the "normal" PCY
itemsFreq = dict()  
hashMap = dict()
hashMapTwo = dict() #second dict
pairsSet =set()
#FIRST STEP

#for every tweet using the flatMap( transforms an RDD of length N into a collection of N collections, then flattens these into a single RDD of results.)
#and than collect returns all elements of a DataFrame  
for line in tqdm(infinityDF.flatMap(lambda tweet : Row(tweet.text.lower().split(" "))).collect()):
    #we create a list to store the not stop words of the tweet
    cleanTweet = set(line) - stopWordsSet
    for word in cleanTweet: #for every word in the tweet's text
        if itemsFreq.get(word):
            itemsFreq[word]+=1
        else: #else create the field.
            itemsFreq[word]=1
    
    couples  = list(itertools.combinations(cleanTweet , 2)) #we use itertools to create all the combination (pairs) in all the words we have of the tweet
    for pair in couples: #for every pair in the combination
        hashedValue  = hash(pair[0] , pair[1] , nextPrime) #we use the 1st hash function to hash the value of the pair to a bucket
        hashedValue2 = secondHash(pair[0] , pair[1] , nextPrime , nextPrime2)
        pairsSet.add(pair)
        if hashMap.get(hashedValue):
            hashMap[hashedValue] += 1
        else:
            hashMap.setdefault(hashedValue , 1)

        if hashMapTwo.get(hashedValue2):
            hashMapTwo[hashedValue2] +=1
        else:
            hashMapTwo.setdefault(hashedValue2 , 1)

print("1:", len(hashMap) ,"   2:",len(hashMapTwo))

bitMap=dict()
bitmapTwo=dict()

for key , value in hashMap.items():
    if value >= support:
        bitMap.setdefault(key , True)
    else:
        bitMap.setdefault(key , False)
del hashMap
for key , value in hashMapTwo.items():
    if value >= support:
        bitmapTwo.setdefault(key , True)
    else:
        bitmapTwo.setdefault(key , True)
del hashMapTwo

frequentPairs=list()
frequentItems=dict()
candidatePairs= dict()

#print(pairsSet)
for pair in pairsSet:
    hkey1 = hash(pair[0] , pair[1] , nextPrime)
    hkey2 = secondHash(pair[0] , pair[1] , nextPrime , nextPrime2)
    #print(bitMap.get(hkey1) and bitmapTwo.get(hkey2))
    if(bitMap.get(hkey1) and bitmapTwo.get(hkey2)):
        if(itemsFreq.get(pair[0] , 0) >= support):
            #frequentItems.append(pair[0])
            if (itemsFreq.get(pair[1] , 0)>= support):
                #frequentItems.append(pair[1])
                candidatePairs.setdefault(pair , 0)
del pairsSet
for line in tqdm(infinityDF.flatMap(lambda tweet : Row(tweet.text.lower().split(" "))).collect()):
    #we create a list to store the not stop words of the tweet
    cleanTweet = set(line) - stopWordsSet
    couples  = list(itertools.combinations(cleanTweet , 2)) #we use itertools to create all the combination (pairs) in all the words we have of the tweet
    for pair in couples: #for every pair in the combination
        if candidatePairs.get(pair) != None:
            candidatePairs[pair] +=1
frequentItems = findMinSupport(itemsFreq, support)
frequentPairs = findMinSupport(candidatePairs, support)

associationRules = list()
for key , value in frequentPairs.items():
    supportPair = float(value) / count
    conf1 = float(value) / frequentItems[key[0]]
    conf2 = float(value) / frequentItems[key[1]]
    associationRules.append(unicode(key[0])+" --> "+unicode(key[1])+" support : "+str(supportPair)+" confidence "+ str(conf1)) 
    associationRules.append(unicode(key[1])+" --> "+unicode(key[0])+" support : "+str(supportPair)+" confidence "+ str(conf2)) 

for elm in associationRules:
    print(elm)

fiList =list()
fpList = list()
for key, value in frequentItems.items():
    temp = [key,value]
    fiList.append(temp)
for key, value in frequentPairs.items():
    temp = [key,value]
    fpList.append(temp)

rdd1 = sc.parallelize(fiList)
rdd2 = sc.parallelize(fpList)

items = rdd1.map(lambda x: Row(item=x[0], frequency=int(x[1])))
pairs = rdd2.map(lambda x: Row(pair=x[0], frequency=int(x[1])))

dataFitems = sqlContext.createDataFrame(items)
dataFpairs = sqlContext.createDataFrame(pairs)

dataFpairs.sort(col('frequency').desc()).show(20)
dataFitems.sort(col('frequency').desc()).show(20)

dataFitems.toPandas().to_csv('/textItems.csv')
dataFpairs.toPandas().to_csv('/textPairs.csv')

#dataFitems.write.format('solr').option("zkHost","172.16.175.155:9983").option("collection","endGame").save()
