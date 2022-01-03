#project to learn how social media algorithm's work
#using "instagram_reach.csv" dataset provided by 'thecleverprogrammer' to complete this learning project
#project will read display the instagram data, train the algorithm, and find out how the
#algorithm differs depending on follower count and 'likes' amounts.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#the data which will be used to examine the instagram algorithm
data_df = pd.read_csv("instagram_reach.csv")
data_df.info()

#check the amount of data in the dataset
display(data_df)

#create a regplot where the followers is the x axis and the amount of likes the y
plt.figure(figsize=(15,10))
plt.grid(True)
plt.title("Regplot for the amount of Likes over how many Followers")
sns.regplot(x = data_df["Followers"], y = data_df["Likes"])
plt.show()

#ensure the data type of all the columns is the same
#'Time since posted' is the only column that was an 'object' data type
data_df['Time since posted'].astype(int)
data_df['Likes'].astype(int)
data_df['Followers'].astype(int)

#regplot where the time is the x axis and the amount of likes the y axis
plt.figure(figsize=(12,10))
plt.grid(True)
plt.title("Regplot for the amount of Likes over how much Time")
sns.regplot(x = data_df["Time since posted"] , y = data_df["Likes"])
plt.show()

#simple training model

features = np.array(data_df[["Followers", "Time since posted"]], dtype = float)
targets = np.array(data_df["Likes"], dtype = float)
likes_val = max(targets)

#displays the largest amount of likes

print(features)
print(targets)
print("Max value of the target is: ",likes_val)

#349.0 is the max value of the target

targets = targets/likes_val
xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.1, random_state = 42)

stdsc = StandardScaler()
xTrain = stdsc.fit_transform(xTrain)
xTest = stdsc.transform(xTest)

gbr = GradientBoostingRegressor()
gbr.fit(xTrain, yTrain)

predictions = gbr.predict(xTest)
plt.scatter(yTest, predictions)
plt.style.use('seaborn-whitegrid')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('GradientRegressor')
plt.plot(np.arange(0,0.4, 0.01), np.arange(0, 0.4, 0.01), color = 'red')
plt.grid(True)
plt.show()

#function to plot the predictions made with the machine learning model above

def PredictionsWithConstantFollowers(model, followerCount, scaller, likes_val):
    followers = followerCount * np.ones(24)
    hours = np.arange(1, 25)

    print(followers)

    #defining vector
    featureVector = np.zeros((24,2))
    featureVector[:,0] = followers
    featureVector[:,1] = hours

    print(featureVector)

    #doing scalling
    featureVector = scaller.transform(featureVector)
    predictions = model.predict(featureVector)
    predictions = (likes_val * predictions).astype('int')

    plt.figure(figsize= (12,12))
    plt.plot(hours, predictions)
    sns.regplot(x= hours, y= predictions)
    plt.grid(True)
    plt.xlabel('hours since posted')
    plt.ylabel('Likes')
    plt.title('Likes progression with ' + str(followerCount) + ' followers')
    plt.show()

#likes progression/prediction with 100 followers
PredictionsWithConstantFollowers(gbr, 100, stdsc, likes_val)

#likes progression/prediction with 200 followers
PredictionsWithConstantFollowers(gbr, 200, stdsc, likes_val)

#likes progression/prediction with 300 followers
PredictionsWithConstantFollowers(gbr, 500, stdsc, likes_val)

#likes progression/prediction with 1000 followers
PredictionsWithConstantFollowers(gbr, 1000, stdsc, likes_val)

