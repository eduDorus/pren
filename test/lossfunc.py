import numpy as np
outputVec = np.array([1,0,0,0,1.5,.4,.6,.7])
history = np.empty_like(outputVec)
step = 1

def loss_funciton(outputVec):
	return np.sum((outputVec - (outputVec > 0.5) * 1 ) ** 2)


def loss_relevanz(vecHistory, step, treshhold=0.01):
	# Bad Outputs
	boolvec = (vecHistory / step) > treshhold
	loss = np.sum((vecHistory - treshhold) * boolvec)
	return loss ** 2 

def make_history(vecCurrent, vecHistory):
	return np.add(vecCurrent, vecHistory)


def loss_gesammt(outputVec):
	global history
	global step
	history = make_history(outputVec, history)
	step = step + 1
	return loss_relevanz(history, step = step) + loss_funciton(outputVec)

loss_gesammt(outputVec)

print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(loss_gesammt(outputVec))
print(step)