import numpy as np  
import pandas as pd  
import csv
from sklearn.ensemble import RandomForestClassifier

def munge_data(df):
    #transform 'Sex' column to numerical values
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    #fill in nans with median ages for each classtr
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Sex'] == i) & \
                                (df['Pclass'] == j+1)]['Age'].dropna().median()

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), \
                    'Age'] = median_ages[i,j]

    #transform 'Embarked' column to numerical
    df['Embarked'] = df['Embarked'].dropna().map( {'C': 1, 'S': 2, 'Q': 3} ).astype(int)
    
    #find mode of 'Embarked' column
    mode = df['Embarked'].dropna().mode().astype(int)
     
    #fill NaNs with mode 
    df['Embarked'] = df['Embarked'].fillna(mode)

    #drop columns with strings that we can't use. 
    df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return df.fillna(0).values

#import dataframe
train_df = pd.read_csv('train.csv', header=0)
test_df  = pd.read_csv('test.csv', header=0)

#save ids from test dataset
ids = test_df['PassengerId'].values

#munge data
train_data = munge_data(train_df)
test_data  = munge_data(test_df)

print('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print('Predicting...')
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()



