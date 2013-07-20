
# File input
r = requests.get(link)
ks =r.json().values()
bucket = 'http://cluster-7-slave-06.sl.hackreduce.net:8098/buckets/dark-stormy-stock-state/keys/'

def bucket_address(bucket, ks):
    bucket_addy = []
    dataDict={}
    for strA in ks:
        bucketVal= bucket + strA
        theData=requests.get(bucketVal)
        # returns a dict with KEY and VALUE
        dataDict[strA]=theData.content        
    return dataDict


    
    
