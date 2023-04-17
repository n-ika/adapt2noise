import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


durs = np.arange(-3,1,0.1)
f1s = np.arange(-3,-0.4,0.1)
eh = np.array(list(product(durs,f1s)))
lbl1 = np.zeros(eh.shape[0])

durs = np.arange(-1,3.1,0.1)
f1s = np.arange(0.5,3.1,0.1)
ah = np.array(list(product(durs,f1s)))
lbl2 = np.ones(ah.shape[0])

lbl = np.concatenate([lbl1,lbl2])
phone = np.concatenate([eh,ah])

df = pd.DataFrame(columns=["duration","f1","phone"])
df["duration"] = phone[:,0]
df["f1"] = phone[:,1]
df["phone"] = lbl

fig, axes = plt.subplots(figsize=(10,10), dpi=100)
sns.scatterplot(data=df, x="duration", y="f1",hue="phone",palette="husl") #.set(title='Buckeye Corpus')
plt.legend([],[], frameon=False)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)
plt.xlabel('Duration (z-score)', fontsize=30)
# plt.xlim([0, 0.5])
plt.ylabel('F1 (z-score)', fontsize=30)
axes.set_title("Assumed data" , size=40)
plt.savefig("./assumed_train.jpg")


corpus="buc"

data_csv = pd.read_csv('./'+corpus+'_vowels_norm.csv')
data_csv['label'] = 0
data_csv.loc[(data_csv.phone == 'AE'), 'label'] = 1

data_csv = data_csv[(data_csv.f1<=1200.0) & (data_csv.duration>=0.03)]

fig, axes = plt.subplots(figsize=(10,10), dpi=100)
sns.scatterplot(data=data_csv, x="norm_duration", y="norm_f1",hue="phone",palette="husl") #.set(title='Buckeye Corpus')
plt.legend([],[], frameon=False)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)
plt.xlabel('Duration (z-score)', fontsize=30)
# plt.xlim([0, 0.5])
plt.ylabel('F1 (z-score)', fontsize=30)
axes.set_title("TIMIT Corpus" , size=40)
plt.savefig("./"+corpus+"_z_train.jpg")

