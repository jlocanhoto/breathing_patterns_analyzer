from sklearn.metrics import classification_report
import numpy as np

# y_test = ['exerc_2','exerc_3','exerc_4','exerc_5','exerc_5','exerc_2','exerc_6','exerc_3','exerc_4','exerc_1','exerc_1','exerc_3','exerc_1','exerc_3','exerc_6','exerc_4','exerc_1','exerc_1','exerc_7','exerc_6','exerc_5','exerc_7','exerc_7','exerc_6','exerc_7','exerc_2','exerc_5','exerc_3','exerc_4','exerc_4','exerc_2','exerc_5','exerc_2','exerc_6','exerc_7']
# y_resp_semi_specialist = ['exerc_7','exerc_3','exerc_4','exerc_5','exerc_5','exerc_2','exerc_7','exerc_3','exerc_4','exerc_7','exerc_1','exerc_3','exerc_6','exerc_1','exerc_6','exerc_4','exerc_1','exerc_1','exerc_1','exerc_6','exerc_5','exerc_1','exerc_7','exerc_6','exerc_7','exerc_6','exerc_5','exerc_3','exerc_4','exerc_6','exerc_2','exerc_5','exerc_2','exerc_6','exerc_6']
# y_resp_layman = ['exerc_7','exerc_3','exerc_4','exerc_5','exerc_5','exerc_2','exerc_6','exerc_3','exerc_4','exerc_7','exerc_1','exerc_3','exerc_7','exerc_3','exerc_6','exerc_4','exerc_7','exerc_1','exerc_7','exerc_6','exerc_5','exerc_2','exerc_7','exerc_7','exerc_7','exerc_7','exerc_5','exerc_3','exerc_4','exerc_6','exerc_7','exerc_5','exerc_7','exerc_6','exerc_6']
# y_resp_specialist = ['exerc_2','exerc_3','exerc_4','exerc_5','exerc_5','exerc_2','exerc_7','exerc_3','exerc_4','exerc_7','exerc_5','exerc_3','exerc_1','exerc_1','exerc_6','exerc_4','exerc_1','exerc_1','exerc_7','exerc_6','exerc_5','exerc_7','exerc_7','exerc_2','exerc_7','exerc_1','exerc_5','exerc_3','exerc_4','exerc_3','exerc_2','exerc_5','exerc_2','exerc_7','exerc_7']

# print(classification_report(y_test, y_resp_specialist, digits=4))

print(np.mean([0.7627, 0.7269, 0.6926]))

'''
semi_specialist

            precision    recall  f1-score   support

     exerc_1     0.5000    0.6000    0.5455         5
     exerc_2     1.0000    0.6000    0.7500         5
     exerc_3     1.0000    0.8000    0.8889         5
     exerc_4     1.0000    0.8000    0.8889         5
     exerc_5     1.0000    1.0000    1.0000         5
     exerc_6     0.5000    0.8000    0.6154         5
     exerc_7     0.4000    0.4000    0.4000         5

    accuracy                         0.7143        35
   macro avg     0.7714    0.7143    0.7269        35
weighted avg     0.7714    0.7143    0.7269        35

'''

'''
layman

              precision    recall  f1-score   support

     exerc_1     1.0000    0.4000    0.5714         5
     exerc_2     0.5000    0.2000    0.2857         5
     exerc_3     1.0000    1.0000    1.0000         5
     exerc_4     1.0000    0.8000    0.8889         5
     exerc_5     1.0000    1.0000    1.0000         5
     exerc_6     0.6667    0.8000    0.7273         5
     exerc_7     0.2727    0.6000    0.3750         5

    accuracy                         0.6857        35
   macro avg     0.7771    0.6857    0.6926        35
weighted avg     0.7771    0.6857    0.6926        35
'''

'''
specialist
              precision    recall  f1-score   support

     exerc_1     0.6000    0.6000    0.6000         5
     exerc_2     0.8000    0.8000    0.8000         5
     exerc_3     0.8000    0.8000    0.8000         5
     exerc_4     1.0000    0.8000    0.8889         5
     exerc_5     0.8333    1.0000    0.9091         5
     exerc_6     1.0000    0.4000    0.5714         5
     exerc_7     0.6250    1.0000    0.7692         5

    accuracy                         0.7714        35
   macro avg     0.8083    0.7714    0.7627        35
weighted avg     0.8083    0.7714    0.7627        35
'''