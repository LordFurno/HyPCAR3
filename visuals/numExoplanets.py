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

# Set figure size explicitly
plt.figure(figsize=(12, 8))  # Width x Height in inches
plt.bar(years,cumulative,color="#007ACC",width=0.8)
plt.title("Cumulative Discovery of Exoplanets Over Time",fontsize=16,fontweight="bold")
plt.xlabel("Discovery Year",fontsize=14)
plt.ylabel("Total Number of Exoplanets",fontsize=14)
plt.grid(axis="y",color="gray",linestyle="--",linewidth=0.5,alpha=0.7)
plt.xticks(years,fontsize=10,rotation=45)
plt.tight_layout()

# Save the figure with the specified DPI for better quality
plt.savefig(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\exoplanetsDiscovered.png", dpi=300)  # Higher DPI for sharp images
plt.show(block=True)