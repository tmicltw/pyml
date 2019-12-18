import numpy as np

w = np.array([0.1, 0.3, -20])

def predict(report_score, exam_score):
    x = [report_score, exam_score, 1]
    if np.matmul(x, w) >= 0:
        return 1
    else:
        return 0

for r, e in [(20, 30), (40, 50), (50, 50), (50, 60), (60, 80), (10, 90), (90, 10)]:
    print("report: {0}, exam: {1}, predicition: {2}".format(r, e, predict(r, e)))
