import pandas as pd 													#impoting librarires
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib


def method_1():	
	print('\nremoving rows with zero as data in corpus')
	data=pd.read_csv('50_Startups.csv')									#storing data and dividing in train and test
	train_data=data.loc[:40]
	test_data=data.loc[41:]
	data=data.replace(0,np.nan)											#removing 0 win nan		
	data=data.dropna(how='any',axis=0)									# removing nan from the data
	x=train_data[['R&D Spend','Administration','Marketing Spend']]		
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)														#training the model
	predi=regr.predict(test_data[['R&D Spend','Administration','Marketing Spend']])			#testing the model
	print('Coefficient of determination: %.2f'%r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)

def method_2():
	print('\nreplacing zero with mean of the column data')
	data=pd.read_csv('50_Startups.csv')										#storing data and dividing in train and test
	data['R&D Spend']=data['R&D Spend'].replace(0,pd.DataFrame.mean(data['R&D Spend']))		#replacing 0 with the mean of
	data['Marketing Spend']=data['Marketing Spend'].replace(0								#all the data 	
											,pd.DataFrame.mean(data['Marketing Spend']))	# in that column
	train_data=data.loc[:40]												
	test_data=data.loc[41:]
	x=train_data[['R&D Spend','Administration','Marketing Spend']]
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)
	predi=regr.predict(test_data[['R&D Spend','Administration','Marketing Spend']])
	print('Coefficient of determination: %.2f'%r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)

def method_3():
	print('\nusing column R&D Spend and Administration to train the model')
	data=pd.read_csv('50_Startups.csv')									#storing data and dividing in train and test
	train_data=data.loc[:40]
	test_data=data.loc[41:]
	x=train_data[['R&D Spend','Administration']]									#using only 2 columns
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)																			#training the model		
	predi=regr.predict(test_data[['R&D Spend','Administration']])							#testing
	print('Coefficient of determination: %.2f'%r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)

def method_4():
	print('\nusing Administration and Marketing Spend')
	data=pd.read_csv('50_Startups.csv')								#storing data and dividing in train and test
	train_data=data.loc[:40]
	test_data=data.loc[41:]
	x=train_data[['Administration','Marketing Spend']]						#using only 2 columns
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)															#training the model		
	predi=regr.predict(test_data[['Administration','Marketing Spend']])		#testing the model	
	print('Coefficient of determination: %.2f'%r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)	

def method_5():
	print('\nusing column R&D Spend and Marketing Spend')
	data=pd.read_csv('50_Startups.csv')								#storing data and dividing in train and test
	train_data=data.loc[:30]
	test_data=data.loc[31:]
	x=train_data[['R&D Spend','Marketing Spend']]							#using only 2 columns
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)																			#training			
	predi=regr.predict(test_data[['R&D Spend','Marketing Spend']])							#testing
	print('Coefficient of determination: %.2f'%r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)


def method_6():
	print('\nusing dummies of states to train')
	data1=pd.read_csv('50_Startups.csv') 							#storing data and dividing in train and test
	dummies=pd.get_dummies(data1.State)								#creating dummies
	merged=pd.concat([data1,dummies],axis='columns')				#joining dummies with original data
	data=merged.drop(['State','California'],axis=1)					#droping the unecessary columns
	train_data=data.loc[11:]
	test_data=data.loc[:10]
	x=train_data[['R&D Spend','Administration','Marketing Spend','Florida','New York']]
	y=train_data[['Profit']]
	regr=linear_model.LinearRegression()
	regr.fit(x,y)																						#training
	predi=regr.predict(test_data[['R&D Spend','Administration','Marketing Spend','Florida','New York']]) #testing
	print('Coefficient of determination: %.2f' %r2_score(test_data['Profit'], predi))		# calculating r^2
	pred=r2_score(test_data['Profit'], predi)
	return round(pred,2)

def method_7():
	print('\nusing train_test_split to split the data')
	data=pd.read_csv('50_Startups.csv')                                 #storing data and dividing in train and test
	y=data.Profit
	x=data.drop('Profit',axis=1)
	x=x.drop('State',axis=1)
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
	# regr=linear_model.LinearRegression()
	# regr.fit(x_train,y_train)  						#training the model
	# joblib.dump(regr,'model_joblib')					#saving the model			
	model=joblib.load('model_joblib')                   #extracting the model        
	predi=model.predict(x_test)                         #testing
	print('Coefficient of determination: %.2f'%r2_score(y_test, predi))
	pred=r2_score(y_test, predi)
	return round(pred,2)


predi1=method_1()
input()
predi2=method_2()
input()
predi3=method_3()
input()
predi4=method_4()
input()
predi5=method_5()
input()
predi6=method_6()
input()
predi7=method_7()
input()
table=pd.DataFrame()
table['r2 value ']=[predi1,predi2,predi3,predi4,predi5,predi6,predi7]
print('\n',table)