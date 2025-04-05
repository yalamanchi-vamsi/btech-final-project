# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
def association(l,data):
        dfObj=pd.DataFrame(data)
        #print(l)
        cols=['LGT_COND','WEATHER','SUR_COND','MAN_COLL','DRUNK_DR','FATALS','PERSONS','STATE']
        df = pd.DataFrame(data[cols])
        df['RATE'] = df.apply(lambda row:"HIGH" if (row.FATALS/row.PERSONS)>0.4 else "LOW" ,axis=1) 
        df['BOOL']=df.STATE.isin(l)
        dfObj=df[df['BOOL']==True]
        #print(dfObj)
        dfObj.drop(['FATALS', 'PERSONS','STATE'], axis = 1,inplace=True) 
        apriori_alg(dfObj)
def apriori_alg(dfObj):
        dfObj.loc[(dfObj.LGT_COND == 1),'LGT_COND']='DayLight' 
        dfObj.loc[(dfObj.LGT_COND == 2),'LGT_COND']='Dark' 
        dfObj.loc[(dfObj.LGT_COND == 3),'LGT_COND']='Dark but Lighted' 
        dfObj.loc[(dfObj.LGT_COND == 4),'LGT_COND']='Dawn' 
        dfObj.loc[(dfObj.LGT_COND == 5),'LGT_COND']='Dusk' 
        dfObj.loc[(dfObj.LGT_COND == 9),'LGT_COND']='Unknown'
        ##print(dfObj['LGT_COND'].unique(),data['LGT_COND'].value_counts()) 
        dfObj.loc[(dfObj.WEATHER  == 1),'WEATHER']='Clear/Cloudy'
        dfObj.loc[(dfObj.WEATHER  == 2),'WEATHER']='Rain'
        dfObj.loc[(dfObj.WEATHER  == 3),'WEATHER']='Sleet(hail)'
        dfObj.loc[(dfObj.WEATHER  == 4),'WEATHER']='Snow or blowing snow' 
        dfObj.loc[(dfObj.WEATHER  == 5),'WEATHER']='Fog/Smoke/Smog' 
        dfObj.loc[(dfObj.WEATHER  == 6),'WEATHER']='Severe Crosswinds'
        dfObj.loc[(dfObj.WEATHER  == 7),'WEATHER']='Blowing sand,soil,dirt'
        dfObj.loc[(dfObj.WEATHER  == 8),'WEATHER']='Other'
        dfObj.loc[(dfObj.WEATHER  == 9),'WEATHER']='Unknown'
        ##print(dfObj['WEATHER'].unique(),data['WEATHER'].value_counts())
        
        dfObj.loc[(dfObj.SUR_COND  == 1),'SUR_COND']='Dry'
        dfObj.loc[(dfObj.SUR_COND  == 2),'SUR_COND']='Wet'
        dfObj.loc[(dfObj.SUR_COND  == 3),'SUR_COND']='Snow/Slush'
        dfObj.loc[(dfObj.SUR_COND  == 4),'SUR_COND']='Ice/Frost' 
        dfObj.loc[(dfObj.SUR_COND  == 5),'SUR_COND']='Sand/Dirt/Mud/Gravel' 
        dfObj.loc[(dfObj.SUR_COND  == 6),'SUR_COND']='Water[standing or moving]'
        dfObj.loc[(dfObj.SUR_COND  == 7),'SUR_COND']='Oil'
        dfObj.loc[(dfObj.SUR_COND  == 8),'SUR_COND']='Other'
        dfObj.loc[(dfObj.SUR_COND  == 9),'SUR_COND']='UnknoZZwn'
        
        dfObj.loc[(dfObj.MAN_COLL  ==0),'MAN_COLL']='No collision with vehicle in transport'
        dfObj.loc[(dfObj.MAN_COLL  ==1),'MAN_COLL']='Front-to-Rear'
        dfObj.loc[(dfObj.MAN_COLL  == 2),'MAN_COLL']='Front-to-Front'
        dfObj.loc[(dfObj.MAN_COLL  == 3),'MAN_COLL']='Front-to-Side(Same direction)'
        dfObj.loc[(dfObj.MAN_COLL  == 4),'MAN_COLL']='Front-to-Side(Opposite direction)' 
        dfObj.loc[(dfObj.MAN_COLL  == 5),'MAN_COLL']='Front-to-Side(Right angle)' 
        dfObj.loc[(dfObj.MAN_COLL  == 6),'MAN_COLL']='Front-to-Side(Angle not specified)'
        dfObj.loc[(dfObj.MAN_COLL  == 7),'MAN_COLL']='Sideswipe(Same direction)'
        dfObj.loc[(dfObj.MAN_COLL  == 8),'MAN_COLL']='Sideswipe(Opposite direction)'
        dfObj.loc[(dfObj.MAN_COLL  == 9),'MAN_COLL']='Rear-to-Side'
        dfObj.loc[(dfObj.MAN_COLL  == 10),'MAN_COLL']='Rear-to-Rear'
        dfObj.loc[(dfObj.MAN_COLL  == 11),'MAN_COLL']='Other'
        dfObj.loc[(dfObj.MAN_COLL  == 99),'MAN_COLL']='Unknown'
        

        dfObj.loc[(dfObj.DRUNK_DR  == 0),'DRUNK_DR']='No'
        dfObj.loc[(dfObj.DRUNK_DR  == 1),'DRUNK_DR']='Yes'
        dfObj.loc[(dfObj.DRUNK_DR  == 2 ),'DRUNK_DR']='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 3 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 4 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 5 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 6 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 7 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 8 )]='Unknown'
        dfObj.loc[(dfObj.DRUNK_DR  == 9 )]='Unknown'
        records = []
        l=len(dfObj)        
        print(l)
        for i in range(0, l):
            records.append([str(dfObj.values[i,j]) for j in range(0, 6)])
        te=TransactionEncoder()
        te_ary=te.fit(records).transform(records)
        df1=pd.DataFrame(te_ary,columns=te.columns_)
        frequent_items=apriori(df1,min_support=0.3,use_colnames=True)
        #print(len(frequent_items))
        frequent_items['length']=frequent_items['itemsets'].apply(lambda x: len(x))
        #print(frequent_items)
        rules=association_rules(frequent_items,metric="lift",min_threshold=1)
        #print(len(rules))
        cols=['antecedents','consequents','confidence']
        #rules_f1=rules[(rules['confidence']>=0.5) & (rules['consequents']=={'LOW'} ) ]
        #print(rules_f1[cols])
        rules_f2=rules[(rules['confidence']>=0.5) & (rules['consequents']=={'HIGH'}) ]
        print(rules_f2[cols])
def classification(data):
    print("Classification to predicted  class label for unknown data:\n")
    df=pd.DataFrame(data)
    cols=['LGT_COND','WEATHER','SUR_COND','MAN_COLL','DRUNK_DR','FATALS','PERSONS']
    dfObj = pd.DataFrame(df[cols])
    dfObj['FRATE'] = dfObj.apply(lambda row: row.FATALS/row.PERSONS, axis = 1) 
    dfObj['RATE'] = dfObj.apply(lambda row:1 if row.FRATE>0.4 else 0 , axis = 1) 
    #high=1;low=0
    dfObj.drop(['FATALS', 'PERSONS','FRATE'], axis = 1,inplace=True) 
    x=dfObj.iloc[:,0:5].values
    y=dfObj.iloc[:,5].values
    print("ENTER DETAILS FOR PREDICTION:")

    w=int(input("Enter Weather condition:\n 1)Clear/Cloudy \n 2)Rain \n 3)Sleet(hail) \n 4)Snow or blowing snow \n 5)Fog/Smoke/Smog \n 6)Severe Crosswinds \n 7)Blowing sand,soil,dirt \n 8)Other \n 9)Unknown"))
    s=int(input("Enter Suraface Condition:\n 1)Dry \n 2)Wet \n 3)Snow/Slush \n 4)Ice/Frost \n 5)Sand/Dirt/Mud/Gravel \n 6)Water[standing or moving] \n 7)Oil \n 8)Other \n 9)Unknown"))
    d=int(input("Enter Drunk state:\n 0)Yes \n 1)No\n 2)Unknown"))
    l=int(input(" Enter Light condition:\n 1)DayLight \n 2)Dark\n 3)Dark but Lighted \n 4)Dawn \n 5)Dusk \n 9)Unknown"))
    m=int(input("Enter Manner of Collision:\n 0)No collision with vehicle in transport\n 1)Front-to-Rear\n 2)Front-to-Front\n 3)Front-to-Side(Same direction)\n 4)Front-to-Side(Opposite direction)\n 5)Front-to-Side(Right angle) \n 6)Front-to-Side(Angle not specified)\n 7)Sideswipe(Same direction)\n 8)Sideswipe(Opposite direction)\n 9)Rear-to-Side\n 10)Rear-to-Rear\n 11)Other\n 99)Unknown"))
    li=[]
    li.append(l)
    li.append(w)
    li.append(s)
    li.append(m)
    li.append(d)
    gnb = GaussianNB()
    gnb.fit(x, y)
    y_pred = gnb.predict([li])
    if(y_pred[0]==0):
        print("The predicted class label for given data is - LOW")
    else:
        print("The predicted class label for given data is - HIGH")
    #print(y_pred)
    #Model accuracy
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.01,random_state=0) # 70% training and 30% testing
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Accuracy of the classification is :",metrics.accuracy_score(y_test, y_pred))
 
def counties_analysis(data):
    df=pd.DataFrame(data)
    cols=['STATE','SP_LIMIT','SUR_COND','DRUNK_DR','WEATHER','LGT_COND','FATALS','COUNTY','PERSONS']
    dfObj = pd.DataFrame(df[cols])
    indexNames = dfObj[ dfObj['SP_LIMIT'] == 99].index
    dfObj.drop(indexNames,inplace=True )
    dfObj.loc[(dfObj.STATE  == 1),'STATENAME']='Alabama'
    dfObj.loc[(dfObj.STATE  == 2),'STATENAME']='Alaska'
    #dfObj.loc[(dfObj.STATE  == 3),'STATE1']=''
    dfObj.loc[(dfObj.STATE  == 4),'STATENAME']='Arizona'
    dfObj.loc[(dfObj.STATE  == 5),'STATENAME']='Arkansas'
    dfObj.loc[(dfObj.STATE  == 6),'STATENAME']='California'
    dfObj.loc[(dfObj.STATE  == 7),'STATENAME']=''
    dfObj.loc[(dfObj.STATE  == 8),'STATENAME']='Colorado'
    dfObj.loc[(dfObj.STATE  == 9),'STATENAME']='Connecticut'
    dfObj.loc[(dfObj.STATE  == 10),'STATENAME']='Delaware'
    dfObj.loc[(dfObj.STATE  == 11),'STATENAME']='District of Columbia'
    dfObj.loc[(dfObj.STATE  == 12),'STATENAME']='Florida'
    dfObj.loc[(dfObj.STATE  == 13),'STATENAME']='Georgia'
    #dfObj.loc[(dfObj.STATE  == 14),'STATE1']=''
    dfObj.loc[(dfObj.STATE  == 15),'STATENAME']='Hawaii'
    dfObj.loc[(dfObj.STATE  == 16),'STATENAME']='Idaho'
    dfObj.loc[(dfObj.STATE  == 17),'STATENAME']='Illinois'
    dfObj.loc[(dfObj.STATE  == 18),'STATENAME']='Indiana'
    dfObj.loc[(dfObj.STATE  == 19),'STATENAME']='Iowa'
    dfObj.loc[(dfObj.STATE  == 20),'STATENAME']='Kansas'
    dfObj.loc[(dfObj.STATE  == 21),'STATENAME']='Kentucky'
    dfObj.loc[(dfObj.STATE  == 22),'STATENAME']='Louisiana'
    dfObj.loc[(dfObj.STATE  == 23),'STATENAME']='Maine'
    dfObj.loc[(dfObj.STATE  == 24),'STATENAME']='Maryland'
    dfObj.loc[(dfObj.STATE  == 25),'STATENAME']='Massachusetts'
    dfObj.loc[(dfObj.STATE  == 26),'STATENAME']='Michigan'
    dfObj.loc[(dfObj.STATE  == 27),'STATENAME']='Minnesota'
    dfObj.loc[(dfObj.STATE  == 28),'STATENAME']='Mississippi'
    dfObj.loc[(dfObj.STATE  == 29),'STATENAME']='Missouri'
    dfObj.loc[(dfObj.STATE  == 30),'STATENAME']='Montana'
    dfObj.loc[(dfObj.STATE  == 31),'STATENAME']='Nebraska'
    dfObj.loc[(dfObj.STATE  == 32),'STATENAME']='Nevada'
    dfObj.loc[(dfObj.STATE  == 33),'STATENAME']='New Hampshire'
    dfObj.loc[(dfObj.STATE  == 34),'STATENAME']='New Jersey'
    dfObj.loc[(dfObj.STATE  == 35),'STATENAME']='New Mexico'
    dfObj.loc[(dfObj.STATE  == 36),'STATENAME']='New York'
    dfObj.loc[(dfObj.STATE  == 37),'STATENAME']='North Carolina'
    dfObj.loc[(dfObj.STATE  == 38),'STATENAME']='North Dakota'
    dfObj.loc[(dfObj.STATE  == 39),'STATENAME']='Ohio'
    dfObj.loc[(dfObj.STATE  == 40),'STATENAME']='Oklahoma'
    dfObj.loc[(dfObj.STATE  == 41),'STATENAME']='Oregon'
    dfObj.loc[(dfObj.STATE  == 42),'STATENAME']='Pennsylvania'
    dfObj.loc[(dfObj.STATE  == 43),'STATENAME']='Puerto Rico'
    dfObj.loc[(dfObj.STATE  == 44),'STATENAME']='Rhode Island'
    dfObj.loc[(dfObj.STATE  == 45),'STATENAME']='South Carolina'
    dfObj.loc[(dfObj.STATE  == 46),'STATENAME']='South Dakota'
    dfObj.loc[(dfObj.STATE  == 47),'STATENAME']='Tennessee'
    dfObj.loc[(dfObj.STATE  == 48),'STATENAME']='Texas'
    dfObj.loc[(dfObj.STATE  == 49),'STATENAME']='Utah'
    dfObj.loc[(dfObj.STATE  == 50),'STATENAME']='Vermont'
    dfObj.loc[(dfObj.STATE  == 51),'STATENAME']='Virginia'
    dfObj.loc[(dfObj.STATE  == 52),'STATENAME']='Virgin Islands'
    dfObj.loc[(dfObj.STATE  == 53),'STATENAME']='Washington'
    dfObj.loc[(dfObj.STATE  == 54),'STATENAME']='West Virginia'
    dfObj.loc[(dfObj.STATE  == 55),'STATENAME']='Wisconsin'
    dfObj.loc[(dfObj.STATE  == 56),'STATENAME']='Wyoming'
    cols=['STATE','STATENAME']
    s1=input("Enter STATE NAME:")
    county=input("Enter COUNTY NAME:")
    s,c1=fun(s1,county)
    
    if(s=='null'):
        print("Incorrect State Name")
        return
    if(c1=='null'):
        print("incorrect County Name") 
        return
    
    df1=df[df['COUNTY']==c1]
    cols=['LGT_COND','WEATHER','SUR_COND','MAN_COLL','DRUNK_DR','FATALS','PERSONS','STATE']
    df1= df1[cols]
    df1['RATE'] = df1.apply(lambda row:"HIGH" if (row.FATALS/row.PERSONS)>0.5 else "LOW" ,axis=1) 
    df1.drop(['FATALS', 'PERSONS','STATE'], axis = 1,inplace=True) 
    #apriori_alg(df1)
    
     
    if(s!='null' and c1!='null') :
        t1=dfObj[dfObj['STATE']==s]
        t=t1[['COUNTY','PERSONS', 'FATALS']].groupby(['COUNTY'], as_index='COUNTY').sum().sort_values(by='COUNTY', ascending=True)
        cols=['PERSONS','FATALS']
        x=t[cols].copy()
        kmeans = KMeans(3)  
        kmeans.fit(x)
        clusters=x.copy()
        clusters['pred_label']=kmeans.fit_predict(x)
        centroids = kmeans.cluster_centers_
        #print(centroids)
        #labels = kmeans.labels_
        #print(labels)
        plt.scatter(clusters['PERSONS'],clusters['FATALS'],c=clusters['pred_label'],cmap='rainbow')
        plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x" ,s=100)
        plt.xlabel('PERSONS')
        plt.ylabel('FATALS')
        print("CLUSTERING OF COUNTIES IN STATE BASED ON FATALITY:",s1)
        plt.show()
        d={}
        c=[]
        d[0]=centroids[0][1]/centroids[0][0]
        d[1]=centroids[1][1]/centroids[1][0]
        d[2]=centroids[2][1]/centroids[2][0]
        #print(d)
        for key, value in sorted(d.items(), key=lambda item: item[1]):
            c.append(key)
            #print(c)
            #print(clusters)
        clusters.loc[(clusters.pred_label == c[0]),'LABEL']='Safe Zone' 
        clusters.loc[(clusters.pred_label == c[1]),'LABEL']='Risk Zone' 
        clusters.loc[(clusters.pred_label == c[2]),'LABEL']='High Risk Zone' 
        #print(clusters)
        if(c1!='null'):
            print("The county ",county,"is:",clusters.loc[c1,'LABEL'])
        print("frequen item-sets for county",county)
        apriori_alg(df1)
def fun(s,c=9999):
    data=pd.read_csv(r'C:\Users\DELL\Desktop\New folder\COUNTY.csv',encoding='latin-1')
    df=pd.DataFrame(data)
    cols=['State Name','State Code','County Code','County Name']
    dfObj = pd.DataFrame(df[cols])
    if(c!=9999):
        #s=input("Enter STATE NAME:")
        #c=input("Enter COUNTY NAME:")
        sno=dfObj[dfObj['State Name']==s.strip().upper() ]
        if(len(sno)==0):
            return 'null','null'
        else:
            state_no=sno['State Code'].unique()
            df1=sno[sno['County Name']==c.strip().upper()]
            county_no=df1['County Code'].unique()
        if(len(county_no)==0):
            return state_no[0],'null'
        else:
            return state_no[0],county_no[0]  
    else:
        sno=dfObj[dfObj['State Name']==s.strip().upper() ]
        if(len(sno)==0):
            return 'null'
        else:
            state_no=sno['State Code'].unique()
        return state_no[0]
def state_analysis(data):
    df=pd.DataFrame(data)
    state=input("Enter State Name:")
    s=fun(state)
    print("graphical analysis of the state",state)
    graphs(df,s)
    #correlation(df,s)

def graphs(df,s):
    dfObj=df[df['STATE']==s]
    speed_graph(dfObj)
    mcoll_graph(dfObj)
    surc_graph(dfObj)
    lightc_graph(dfObj)
    w_graph(dfObj)
    #speed

def speed_graph(dfObj):
    plt.rcParams["figure.figsize"] = (8,5)
    cols=['SP_LIMIT','FATALS','PERSONS']
    dfObj = pd.DataFrame(dfObj[cols])
    indexNames=dfObj[ dfObj['SP_LIMIT']==99].index
    dfObj.drop(indexNames,inplace=True )
    dataf=dfObj.groupby(['SP_LIMIT']).sum().sum(level=['SP_LIMIT']).reset_index()
    count=dataf.FATALS.sum()
    #print(count)
    dataf['FATAL_PERCENTAGE_SPEED']=dataf.apply(lambda row:(row.FATALS/count)*100,axis=1)
    #print(dataf) 
    plt.xlim((0,99))
    plt.ylim((0,35))
    plt.yticks(range(0, 40, 5))
    plt.xticks(range(0,100,5))
    plt.xlabel("Speed Limit")
    plt.ylabel("Percentage(%)")
    plt.title("Fatal accidents on different speeds")
    plt.bar(dataf['SP_LIMIT'], dataf['FATAL_PERCENTAGE_SPEED'], align='center',color='red',width=3)
    plt.show()   
def mcoll_graph(dfObj):  
    plt.rcParams["figure.figsize"] = (30,5)
    cols=['MAN_COLL','FATALS','PERSONS']
    dfObj4=pd.DataFrame(dfObj[cols])
    indexNames4= dfObj4[dfObj4['MAN_COLL']==99].index
    dfObj4.drop(indexNames4,inplace=True )
    dataf4=dfObj4.groupby(['MAN_COLL']).sum().sum(level=['MAN_COLL']).reset_index()
    count4=dataf4.FATALS.sum()
    #print(count4)
    dataf4['FATAL_PERCENTAGE_COLLISON']=dataf4.apply(lambda row:(row.FATALS/count4)*100,axis=1)
    #print(dataf4)
    plt.xlim((0,12))
    plt.ylim((0,70))
    m=['Not a Collision',
        'FtoR(includes Rear-End)',
        'FtoF(includes Head-On)',
        'FtoS,Same Direction',
        'FtoS,Opposite Direction',
        'FtoS,Right Angle',
        'FtoS(Direction Not Specified)',
        'Sideswipe–Same Direction',
        'Sideswipe–Opposite Direction',
        'RtoS',
        'RtoR',
        'Other(End-Swipes and Others)']
    plt.yticks(range(0,70,10))
    plt.xticks(range(0,12,1),m)
    plt.xlabel("Manner Of Collision")
    plt.ylabel("Percentage(%)")
    plt.title("Fatal accidents on different collision manners")
    plt.bar(dataf4['MAN_COLL'],dataf4['FATAL_PERCENTAGE_COLLISON'],align='center',color='orange',width=0.75)
    plt.show()
def surc_graph(dfObj):
    plt.rcParams["figure.figsize"] = (22,5)
    cols3=['SUR_COND','FATALS','PERSONS']
    dfObj3=pd.DataFrame(dfObj[cols3])
    dataf3=dfObj3.groupby(['SUR_COND']).sum().sum(level=['SUR_COND']).reset_index()
    count3=dataf3.FATALS.sum()
    #print(count3)
    dataf3['FATAL_PERCENTAGE_SURFACE']=dataf3.apply(lambda row:(row.FATALS/count3)*100,axis=1)
    #print(dataf3)
    plt.xlim((0,10))
    plt.ylim((0,100))
    s=['','Dry','Wet','Snow/Slush','Ice/Frost','Sand,Dirt,Mud,Gravel','Water(standing/moving)','Oil','Other','Unknown']
    plt.yticks(range(0,100,10))
    plt.xticks(range(0,10,1),s)
    plt.xlabel("Surface Conditions")
    plt.ylabel("Percentage(%)")
    plt.title("(e)Fatal accidents on different surface conditions")
    plt.bar(dataf3['SUR_COND'],dataf3['FATAL_PERCENTAGE_SURFACE'],align='center',color='VIOLET',width=0.4)
    plt.show()
def lightc_graph(dfObj):
    plt.rcParams["figure.figsize"] = (8,5)
    cols1=['LGT_COND','FATALS','PERSONS']
    dfObj1=pd.DataFrame(dfObj[cols1])
    dataf1=dfObj1.groupby(['LGT_COND']).sum().sum(level=['LGT_COND']).reset_index()
    count1=dataf1.FATALS.sum()
    #print(count1)
    dataf1['FATAL_PERCENTAGE_LIGHT']=dataf1.apply(lambda row:(row.FATALS/count1)*100,axis=1)
    #print(dataf1)
    plt.xlim((0,7))
    plt.ylim((0,65))
    l=['','Daylight','Dark','Dark but lighted','Dawn','Dusk','Unknown']
    plt.yticks(range(0,65,10))
    plt.xticks(range(0,7,1),l)
    plt.xlabel("LIGHT CONDITION")
    plt.ylabel("Percentage(%)") 
    plt.title("Fatal accidents on different light conditions")
    plt.bar(dataf1['LGT_COND'],dataf1['FATAL_PERCENTAGE_LIGHT'],align='center',color='blue',width=0.3)
    plt.show()
def w_graph(dfObj):
    plt.rcParams["figure.figsize"] = (20,6)
    cols2=['WEATHER','FATALS','PERSONS']
    dfObj2=pd.DataFrame(dfObj[cols2])
    dataf2=dfObj2.groupby(['WEATHER']).sum().sum(level=['WEATHER']).reset_index()
    count2=dataf2.FATALS.sum()
    #print(count2)
    dataf2['FATAL_PERCENTAGE_WEATHER']=dataf2.apply(lambda row:(row.FATALS/count2)*100,axis=1)
    #print(dataf2)
    plt.xlim((0,10))
    plt.ylim((0,105))
    w=['','Clear/Cloudy','Rain','Sleet(Hail)','Snow/Blowing Snow','Fog,Smog,Smoke',' Severe Crosswinds','Blowing Sand,Soil,Dirt','Other','Unknown']
    plt.yticks(range(0,105,10))
    plt.xticks(range(0,10,1),w)
    plt.xlabel("Weather Conditions")
    plt.ylabel("Percentage(%)")
    plt.title("(d)Fatal accidents on different weather conditions") 
    plt.bar(dataf2['WEATHER'],dataf2['FATAL_PERCENTAGE_WEATHER'],align='center',color='green',width=0.5)
    plt.show()

def correlation(df,s):
    df=df[df['STATE']==s]
    indexNames = df[ df['HOUR'] == 99].index
    df.drop(indexNames,inplace=True )
    indexNames = df[ df['ARR_HOUR'] == 99].index
    df.drop(indexNames,inplace=True )
    df['ARRIVAL_TIME']=df.apply(lambda row:((row.ARR_HOUR*60+row.ARR_MIN)-(row.HOUR*60+row.MINUTE)) if (row.ARR_HOUR>=row.HOUR) else (((row.ARR_HOUR+24)*60+row.ARR_MIN)-(row.HOUR*60+row.MINUTE)) , axis = 1)
    df['RATE'] = df.apply(lambda row:row.FATALS/row.PERSONS,axis=1) 
    #cols=['HOUR','MINUTE','ARR_HOUR','ARR_MIN','ARRIVAL_TIME','RATE']
    n=df['ARRIVAL_TIME'].corr(df['RATE'])
    if(n>-1 and n<-0.7):
        print("perfect -ve linear relationship")
    if(n>-0.7 and n<-0.5):
        print("Strong -ve linear relationship")
    if(n>-0.5 and n<-0.3):
        print("moderate -ve linear relationship")
    if(n>-0.3 and n<0):
        print("weak -ve linear relationship")
    if(n==0):
        print("no linear relationship")
    if(n>0.7 and n<1):
        print("perfect +ve linear relationship")
    if(n>0.5 and n<0.7):
        print("Strong +ve linear relationship")
    if(n>0.3 and n<0.5):
        print("moderate +ve linear relationship")
    if(n>0 and n<0.3):
        print("weak +ve linear relationship")
    print("Correlation between EMS and FATAL-RATE in the state is",n)

"Main method"
if __name__== '__main__' :      
    data=pd.read_csv(r'C:\Users\DELL\Desktop\New folder\ACCIDENT.csv',encoding='utf-8')
    dfObj=pd.DataFrame(data)
    dfObj=dfObj[['STATE','FATALS']].groupby(['STATE'], as_index='STATE').sum().sort_values(by='STATE', ascending=True)
   #STATEP MEANS STATE POPULATION
    dfObj.loc[(dfObj.index   == 1),'STATEP']=4672840
    dfObj.loc[(dfObj.index  == 2),'STATEP']=680300
    dfObj.loc[(dfObj.index  == 4),'STATEP']=6167681
    dfObj.loc[(dfObj.index  == 5),'STATEP']=2848650
    dfObj.loc[(dfObj.index  == 6),'STATEP']=36250311
    dfObj.loc[(dfObj.index  == 8),'STATEP']=4803868
    dfObj.loc[(dfObj.index  == 9),'STATEP']=3527270
    dfObj.loc[(dfObj.index  == 10),'STATEP']=871749
    dfObj.loc[(dfObj.index  == 11),'STATEP']=574404
    dfObj.loc[(dfObj.index  == 12),'STATEP']=18367842
    dfObj.loc[(dfObj.index  == 13),'STATEP']=9349988
    dfObj.loc[(dfObj.index  == 15),'STATEP']=1315675
    dfObj.loc[(dfObj.index  == 16),'STATEP']=1505105
    dfObj.loc[(dfObj.index  == 17),'STATEP']=12695866
    dfObj.loc[(dfObj.index  == 18),'STATEP']=6379599
    dfObj.loc[(dfObj.index  == 19),'STATEP']=2999212
    dfObj.loc[(dfObj.index  == 20),'STATEP']=2783785
    dfObj.loc[(dfObj.index  == 21),'STATEP']=4256672
    dfObj.loc[(dfObj.index  == 22),'STATEP']=4375581
    dfObj.loc[(dfObj.index  == 23),'STATEP']=1327040
    dfObj.loc[(dfObj.index  == 24),'STATEP']=5653408
    dfObj.loc[(dfObj.index  == 25),'STATEP']=6431559
    dfObj.loc[(dfObj.index  == 26),'STATEP']=10001284
    dfObj.loc[(dfObj.index  == 27),'STATEP']= 5207203
    dfObj.loc[(dfObj.index  == 28),'STATEP']= 2928350
    dfObj.loc[(dfObj.index  == 29),'STATEP']=5887612
    dfObj.loc[(dfObj.index  == 30),'STATEP']= 964706
    dfObj.loc[(dfObj.index  == 31),'STATEP']= 1783440
    dfObj.loc[(dfObj.index  == 32),'STATEP']= 2601072
    dfObj.loc[(dfObj.index  == 33),'STATEP']= 1312540
    dfObj.loc[(dfObj.index  == 34),'STATEP']= 8677885
    dfObj.loc[(dfObj.index  == 35),'STATEP']= 1990070
    dfObj.loc[(dfObj.index == 36),'STATEP']= 19132335
    dfObj.loc[(dfObj.index  == 37),'STATEP']= 9118037
    dfObj.loc[(dfObj.index  == 38),'STATEP']=652822
    dfObj.loc[(dfObj.index  == 39),'STATEP']=11500468
    dfObj.loc[(dfObj.index  == 40),'STATEP']= 3634349
    dfObj.loc[(dfObj.index  == 41),'STATEP']= 3722417
    dfObj.loc[(dfObj.index  == 42),'STATEP']= 12563937
    dfObj.loc[(dfObj.index  == 44),'STATEP']=1057315
    dfObj.loc[(dfObj.index  == 45),'STATEP']= 4444110
    dfObj.loc[(dfObj.index  == 46),'STATEP']= 791623
    dfObj.loc[(dfObj.index  == 47),'STATEP']= 6175727
    dfObj.loc[(dfObj.index  == 48),'STATEP']= 23831983
    dfObj.loc[(dfObj.index  == 49),'STATEP']= 2597746
    dfObj.loc[(dfObj.index  == 50),'STATEP']= 623481
    dfObj.loc[(dfObj.index  == 51),'STATEP']= 7751000
    dfObj.loc[(dfObj.index  == 53),'STATEP']= 6461587
    dfObj.loc[(dfObj.index  == 54),'STATEP']= 1834052
    dfObj.loc[(dfObj.index  == 55),'STATEP']= 5610775
    dfObj.loc[(dfObj.index  == 56),'STATEP']= 534876
    print(dfObj)
    #dfObj['POPULATION'] = dfObj.apply(lambda row: row.STATEP/100000, axis = 1)
    cols=['STATEP','FATALS']
    x=dfObj[cols].copy()
    kmeans = KMeans(3)  
    kmeans.fit(x)
    clusters=x.copy()
    clusters['pred_label']=kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_
    #print(centroids)
    labels = kmeans.labels_
    #print(labels)
    plt.scatter(clusters['STATEP'],clusters['FATALS'],c=clusters['pred_label'],cmap='rainbow')
    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x" ,s=100)
    plt.xlabel('POPULATION')
    plt.ylabel('FATALS')
    print("clustering of states based on fatal rate:")
    plt.show()
    d={}
    c=[]
    d[0]=centroids[0][1]/(centroids[0][0])
    d[1]=centroids[1][1]/(centroids[1][0])
    d[2]=centroids[2][1]/(centroids[2][0])
    for key, value in sorted(d.items(), key=lambda item: item[1]):
        c.append(key)         
    clusters.loc[(clusters.pred_label == c[0]),'LABEL']='Safe Zone' 
    clusters.loc[(clusters.pred_label == c[1]),'LABEL']='Realatively Risk Zone' 
    clusters.loc[(clusters.pred_label == c[2]),'LABEL']='Risk Zone' 
    # print(clusters)
    l0=[] 
    l1=[]
    l2=[]
    l=len(clusters.index)
    # print(l)
    index=list(clusters.index)
    p_label=list(clusters.pred_label)
    #print(index,p_label)
    for i in range(0,l):
        j=p_label[i]
        if(j==0):
            l0.append(index[i])
        elif(j==1):
            l1.append(index[i])
        else:
            l2.append(index[i])
    #print(l0,l1,l2)
    print("frequent item sets and association rules for three different clusters:")
    association(l0,data)
    association(l1,data)
    association(l2,data)
    classification(data)
    print("COUNTY WISE ANALYSIS")
    counties_analysis(data)
    print("STATE WISE ANALYSIS")
    state_analysis(data)