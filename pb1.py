import numpy as np

w = (0.1, 0.3, -20)

def predict(report_score, exam_score):
    if report_score * w[0] + exam_score * w[1] + w[2] >= 0:
        return 1
    else:
        return 0

for r, e in [(20, 30), (40, 50), (50, 50), (50, 60), (60, 80), (10, 90), (90, 10)]:
    print("report: {0}, exam: {1}, predicition: {2}".format(r, e, predict(r, e)))
