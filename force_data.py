import math
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

Q = 10e-12				#primary charge
ep_r = 20           #relative permittivity of medium X
ep_0 = 8.85e-12     #permittivity of vacuum				
R = 10e-9				#length of medium X  between the charges
k = 1/(4*math.pi*ep_0)

def regression(data):
	X = data[:,:2] 
	y = data[:,2]
	#y  = np.reshape(y, (-1,1))

	model = LinearRegression()
	model.fit(X, y)

	return model

if __name__ == '__main__':
	
	data1 =np.array([])
	data2 =np.array([])
	for q in np.arange(1,101,1)*1e-12:     #test charge
		for r in np.arange(1,101,1)*1e-9:    #length of vacuum between the charges
			force1 = k*Q*q/((math.sqrt(ep_r)*R+r)**2)
			force2= k*Q*q/((R+r)**2)
			data1 = np.concatenate((data1,np.array([q, r, force1])))
			data2 = np.concatenate((data2,np.array([q, r, force2])))

	size = np.size(data1)
	data1 = np.reshape(data1, (int(size/3),3))
	data2 = np.reshape(data2, (int(size/3),3))
	
	model1 = regression(data1)
	model2 = regression(data2)


	print(model1.intercept_, model1.coef_)
	print(model2.intercept_, model2.coef_)


	data_eg = np.array([[10e-12, 40e-9],[25e-12, 50e-9],[50e-12, 30e-9],[80e-12, 60e-9]])  #example data

	tot_force =  model1.predict(data_eg)    #total force corresponding to the data above
	charge_contrib = model2.predict(data_eg)		#contribution of charge
	permi_contrib = model1.predict(data_eg) - model2.predict(data_eg)   #contribution of permittivity

	print("Total force =", tot_force)
	print("Contribution of charge=",charge_contrib)
	print("Contribution of permittivity=", permi_contrib)


	labels = ['D1','D2','D3','D4']
	
	fig,ax = plt.subplots()

	ax.barh(labels, charge_contrib, height=0.5 ,label='charge')
	ax.barh(labels, permi_contrib, height=0.5 ,label='permittivity')
	ax.legend()
	plt.show()




