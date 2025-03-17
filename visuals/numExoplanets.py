import matplotlib.pyplot as plt
import numpy as np
path=r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\PS_2025.03.15_10.57.51.csv"
data={}
with open(path) as f:
    for line in f:
        if line[0]!="#":
            line=line.split(",")
            if line[6].isdigit():
                data[int(line[6])]=data.get(int(line[6]),0)+1

values=[]

low,high=min(data.keys()),max(data.keys())
for year in range(low,high+1):
    if year in data:
        values.append((year,data[year]))
values=list(zip(*values))
years=values[0]
nums=values[1]

cumulative=[nums[0]]
for i in range(1,len(nums)):
    cumulative.append(cumulative[-1]+nums[i])


plt.bar(years,cumulative)
plt.title("Cumulative Number of Exoplanets Found Per Year")
plt.xlabel("Year")
plt.ylabel("Cumulative Number of Exoplanets")
plt.savefig(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\exoplanetsDiscovered.png")
plt.show(block=True)
