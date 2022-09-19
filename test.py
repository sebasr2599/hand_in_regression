import model as md
import numpy as np
params = [0,0,0]
samples = [[1,1],[2,2],[3,3],[4,4],[5,5]]
y = [2,4,6,8,10]

for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i]=  [1]+samples[i]
	else:
		samples[i]=  [1,samples[i]]
samples = samples/np.linalg.norm(samples) 
print("normalaized samples:")
print(samples)

params = md.train(params,samples,y)
print(params)
print("Sample ",samples[0])
print(md.h(params,samples[0]))
# if __name__ == "main":
#     main()
