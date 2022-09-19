import numpy as np
def h(params, sample):
	"""This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis

	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 

	Returns:
		Evaluation of h(x)
	"""
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum

def normalize(samples):
   return samples/np.linalg.norm(samples) 
    
def gradientDescent(params,samples,y, alpha):
    temp = list(params)
    for j in range(len(params)):
        acum =0
        for i in range(len(samples)):
            error = h(params,samples[i]) - y[i]
            acum = acum + error*samples[i][j]
        temp[j] = params[j] - alpha*(1/len(samples))*acum
    return temp

def train(params,samples,samples_y,epochs=1000,learning_rate=0.1):
    i =0
    while True:
        og = list(params)
        params = gradientDescent(params,samples,samples_y,learning_rate)
        # print(params)
        i+=1 
        if(og == params or i == epochs):
            print("Final params ")
            print(params)
            break
    return params

def predict(params,samples_to_predict):
    # Returns a list of y for the given samples
    temp = []
    for i in range(len(samples_to_predict)):
        temp.append(h(params,samples_to_predict[i]))
    
    return temp





