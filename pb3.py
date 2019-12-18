import numpy as np


w = np.array([0.1, 0.3, -20])

def predict(scores):
    scores = np.hstack([data, np.ones((len(data), 1))])
    return (np.matmul(scores, w) > 0).astype(int)



data = np.array(((20,30), (60,75), (80,30), (90,70), (95,90), (40,60), (80,90), (30,40), (25,55), (35,25), (80,45), (50,10), (25,80)))
y = np.array((0,1,0,1,1,1,1,0,0,0,1,0,1))
preds = predict(data)
error_count = sum(np.abs(y - preds))
accuracy = (len(data) - error_count) / len(data)
print("accuracy: {0}".format(accuracy))
