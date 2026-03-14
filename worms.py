import requests as rq
import json
import time

url="http://www.marinespecies.org/rest/AphiaRecordsByName/"

cls=['kingdom', 'phylum', 'class', 'order', 'family', 'genus']

def get_info(name):
    global url,cls
    try:
        
        r=rq.get(url+name)
        print(r)
        while r.status_code!=200 and r.status_code!=204:
            print('retrying')
            time.sleep(2)
            r=rq.get(url+name)
            print(r)
    except Exception as e:
        print(name,'error!!!',e)
        return []

    if r.text=='':
        return []
    
    return json.loads(r.text)
    
    

def lookup(name):
    r=rq.get(url+name)
    print(r)
    dc=json.loads(r.text)
    print('n:',len(dc))
    for i in cls:
        print(i,':',dc[0][i])

def read_file(path=r'C:/Users/pytho/Desktop/mycode/proj/aicomp/Fuyo_YOLO_Dataset/Fuyo_YOLO_Dataset/classes.txt'):
    r=[]
    with open(path,encoding='utf-8') as f:
        s=f.readlines()
        for l in s:
            if l=='\n':
                continue
            chn,eng=l.split('(')
            eng=eng.rsplit(')')[0]
            #print(chn,eng)
            r.append((eng,chn))
    return r

def grab(names_path,json_path):
    global cls,url
    names=read_file(names_path)
    dic={}
    for n in names:
        print('####',n[0],n[1],'####')
        info=get_info(n[0])
        print('n:',len(info))
        if len(info)==0:
            print('not found!!!')
        else :
            for i in cls:
                print(i,':',info[0][i])
            
        dic[n[0]]={}
        dic[n[0]]['chn']=n[1]
        dic[n[0]]['info']=info
        time.sleep(3)
        
    with open(json_path,'w',encoding='utf-8') as f:
        js=json.dump(dic,f)

    

    
    
