import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_true = ['exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_1', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_3', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7']
y_pred_flow = ['exerc_6', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_5', 'exerc_6', 'exerc_2', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_3', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_4', 'exerc_3', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_4', 'exerc_2', 'exerc_2', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_7', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_2', 'exerc_5', 'exerc_2', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_7', 'exerc_2', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_2', 'exerc_2', 'exerc_7']
y_pred_vol = ['exerc_6', 'exerc_3', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_3', 'exerc_3', 'exerc_6', 'exerc_2', 'exerc_3', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_6', 'exerc_3', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_3', 'exerc_3', 'exerc_4', 'exerc_3', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_6', 'exerc_7', 'exerc_4', 'exerc_4', 'exerc_7', 'exerc_4', 'exerc_3', 'exerc_7', 'exerc_4', 'exerc_3', 'exerc_7', 'exerc_4', 'exerc_3', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_4', 'exerc_4', 'exerc_7', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_4', 'exerc_7', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_5', 'exerc_7', 'exerc_7', 'exerc_5', 'exerc_6', 'exerc_5', 'exerc_6', 'exerc_5', 'exerc_5', 'exerc_4', 'exerc_5', 'exerc_3', 'exerc_7', 'exerc_7', 'exerc_4', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_3', 'exerc_3', 'exerc_7', 'exerc_5', 'exerc_3', 'exerc_5', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_4', 'exerc_7', 'exerc_5', 'exerc_7', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_3', 'exerc_5', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_7', 'exerc_3', 'exerc_7', 'exerc_5', 'exerc_3', 'exerc_7', 'exerc_5', 'exerc_7', 'exerc_7']
y_pred_sum = ['exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_2', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_2', 'exerc_6', 'exerc_7', 'exerc_7', 'exerc_6', 'exerc_1', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_3', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_4', 'exerc_3', 'exerc_3', 'exerc_6', 'exerc_6', 'exerc_3', 'exerc_4', 'exerc_4', 'exerc_2', 'exerc_2', 'exerc_4', 'exerc_6', 'exerc_6', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_4', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_5', 'exerc_7', 'exerc_7', 'exerc_5', 'exerc_2', 'exerc_7', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_6', 'exerc_2', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_5', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_7', 'exerc_1', 'exerc_6', 'exerc_6', 'exerc_6', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_2', 'exerc_7', 'exerc_6', 'exerc_5', 'exerc_7', 'exerc_7']

y_true = np.array(y_true)
y_pred_flow = np.array(y_pred_flow)
y_pred_vol = np.array(y_pred_vol)
y_pred_sum = np.array(y_pred_sum)

print("[PRECISION, RECALL, FSCORE] (macro) => FLOW")
print(precision_recall_fscore_support(y_true, y_pred_flow, average=None))
print("[PRECISION, RECALL, FSCORE] (macro) => VOLUME")
print(precision_recall_fscore_support(y_true, y_pred_vol, average=None))
print("[PRECISION, RECALL, FSCORE] (macro) => SUM")
print(precision_recall_fscore_support(y_true, y_pred_sum, average=None))

'''
[PRECISION, RECALL, FSCORE] (macro) => FLOW
(0.479153153798833, 0.37697184136812617, 0.35845072481414453, None)
[PRECISION, RECALL, FSCORE] (macro) => VOLUME
(0.18118785975928833, 0.22010909627008696, 0.1885185230012816, None)
[PRECISION, RECALL, FSCORE] (macro) => SUM
(0.4610151753008896, 0.38402378495257755, 0.3626827857242604, None)
'''