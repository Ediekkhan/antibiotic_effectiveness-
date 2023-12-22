import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
df = pd.read_csv('/home/Kanex/Documents/dr An/ML(SVM)/Large_files/editedDiagnosis.csv')
df.head(40)
