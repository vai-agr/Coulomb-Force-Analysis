import math
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

q = 10e-12				#test charge           
ep_0 = 8.85e-12     	#permittivity of vacuum/free space
R = 10e-9				#length of medium X  between the charges
r = 10e-9				# length of vacuum between the charges	
k = 1/(4*math.pi*ep_0) 

def dataGenerator():
	data =np.array([])
	for Q in np.arange(1,101,1)*1e-12:     
		for ep_r in np.arange(1,101,1):    
			force = k*Q*q/((math.sqrt(ep_r)*R+r)**2)
			data = np.concatenate((data,np.array([Q, 1/ep_r, force])))
	size = np.size(data)
	data = np.reshape(data, (int(size/3),3))
	return data

def regression(data):
	X = data[:,:2] 
	y = data[:,2]
	y  = np.reshape(y, (-1,1))
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)
	print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))
	model = LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	#print("F = ", model.intercept_, " + ", model.coef_[0], "(Q) + ", model.coef_[1], "(1/ep_r)")
	print(model.intercept_ , model.coef_)
	print(r2_score(y_test, y_pred))

	#plt.scatter(y_test,y_pred, s=1)
	#plt.show()
	return model

if __name__ == '__main__':
	f_Data = dataGenerator()
	Model  = regression(f_Data)
	X_eg = np.array([[10e-12,1/25],[20e-12,1/40],[40e-12,1/60],[80e-12,1/20]])  #4 random data points
	Q_contrib = Model.coef_[0][0]*X_eg[:,0]
	ep_contrib = Model.coef_[0][1]*X_eg[:,1]
	print(Q_contrib, ep_contrib)
	print(Model.predict(X_eg))

	labels = ['D1','D2','D3','D4']
	
	fig,ax = plt.subplots()

	ax.bar(labels, Q_contrib, label='charge')
	ax.bar(labels, ep_contrib, bottom=Q_contrib, label='permittivity')
	ax.legend()
	plt.show()