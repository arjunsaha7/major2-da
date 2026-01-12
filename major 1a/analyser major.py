import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load new dataset
data = pd.read_csv("StudentsPerformance.csv")
data.index = range(1, len(data) + 1) # index from 1 inplace of 0 

# rename columns to simple names
data.rename(columns={
    "math score": "Maths",
    "reading score": "Reading",
    "writing score": "Writing",
    "gender": "Gender"
}, inplace=True)

# create fake Name column since csv dont have it
data["Name"] = ["Student_" + str(i+1) for i in range(len(data))]

subjects = ["Maths", "Reading", "Writing"]

print("\nraw data loaded\n")
print(data.head())

# subject wise stats
avg = data[subjects].mean()
maxi = data[subjects].max()
mini = data[subjects].min()

print("\navrage marks of each subject\n")
print(avg)

print("\nmax marks in each subject\n")
print(maxi)

print("\nmin marks in each subject\n")
print(mini)

# student total and average
data["Total"] = np.sum(data[subjects], axis=1)
data["Average"] = np.mean(data[subjects], axis=1)

print("\nstudent perfomance\n")
print(data[["Name","Gender","Total","Average"]].head())

# topper
topper = data.loc[data["Total"].idxmax()]
print("\ntopper student\n")
print("name :", topper["Name"])
print("gender :", topper["Gender"])
print("total marks :", topper["Total"])

# consistency
data["Consistency"] = data[subjects].std(axis=1)

print("\nconsistency score\n")
print(data[["Name","Consistency"]].head())

# subject difficult

subject_avg = data[subjects].mean()
difficulty = 100 - subject_avg

print("\nsubject difficulty index\n")
print(difficulty)

# trend
data["Trend"] = data["Average"].diff()

print("\ntrend of students\n")
print(data[["Name","Trend"]].head())

# attendance faker as csv dont have
 
np.random.seed(0)
data["Attendance"] = np.random.randint(50, 100, size=len(data))

corr = data["Attendance"].corr(data["Average"])
print("\nattendance and marks relation :", corr)

# dropout risk fun
def risk(avg, std):
    if avg < 40 or std > 20:
        return "High Risk"
    elif avg < 60:
        return "Medium Risk"
    else:
        return "Low Risk"
#dropout logic
data["Dropout_Risk"] = data.apply(lambda x: risk(x["Average"], x["Consistency"]), axis=1)

print("\ndropout risk report\n")
print(data[["Name","Average","Consistency","Dropout_Risk"]].head())

# grade fun
def grade(avg):
    if avg >= 80:
        return "A"
    elif avg >= 60:
        return "B"
    elif avg >= 40:
        return "C"
    else:
        return "Fail"

data["Grade"] = data["Average"].apply(grade)

print("\ngrades\n")
print(data[["Name","Average","Grade"]].head())

# graphs

plt.figure(figsize=(8,5))
plt.plot(data["Average"].head(30))
plt.title("student perfomance trend")
plt.ylabel("average marks")
plt.show()

plt.figure(figsize=(8,5))
difficulty.plot(kind="bar")
plt.title("subject difficulty")
plt.ylabel("difficulty level")
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(data["Consistency"])
plt.title("consistancy of students")
plt.ylabel("std deviation")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(data["Name"].head(10), data["Average"].head(10))
plt.xticks(rotation=45)
plt.title("student marks comparision")
plt.ylabel("average marks")
plt.show()