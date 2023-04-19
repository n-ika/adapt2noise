import pandas
from matplotlib import pyplot
import numpy
import os

input_f = "BUC_vowels_norm.csv"
output_d = "BUC"
relevant_vowels = ["AE","EH"]
corpus = "BUC"

db = pandas.read_csv(input_f)

labels = {
    "AE": "æ",
    "EH": "ε"
}

output_dir = f"vowel_charts/{output_d}"
try:
    os.makedirs(output_dir)
except:
    pass

def meta(title, xaxis, yaxis):
    pyplot.title(title)
    pyplot.ylabel(yaxis)
    pyplot.xlabel(xaxis)

def histogram(column, bins, title, xaxis):
    for v in relevant_vowels:
        points = db[column][db["phone"] == v].map(float).tolist()
        print(points)
        pyplot.hist(points, bins, alpha=0.5, label=labels[v])
    pyplot.legend(loc='upper right')
    meta(title, xaxis, "Count")
    pyplot.savefig(f"{output_dir}/{column}.png")
    pyplot.clf()

def scatter(column1, column2, title, xaxis, yaxis):
    s = pyplot.rcParams['lines.markersize'] ** 1
    for v in relevant_vowels:
        c1 = db[column1][db["phone"] == v].map(float).tolist()
        c2 = db[column2][db["phone"] == v].map(float).tolist()
        pyplot.scatter(c1, c2, s=s, alpha=0.5, label=labels[v])
    meta(title, xaxis, yaxis)
    pyplot.legend(loc='upper right')
    pyplot.savefig(f"{output_dir}/{column1}-against-{column2}.png")
    pyplot.clf()

# f1
histogram("norm_f1", numpy.linspace(-3,3,100), f"Speaker-Normalized F1, {corpus}", "z-score")

# f2
histogram("norm_f2", numpy.linspace(-3,3,100), f"Speaker-Normalized F2, {corpus}", "z-score")

# f2 - f1
histogram("norm_deltaFormant", numpy.linspace(-3,3,100), f"Speaker-Normalized F2-F1, {corpus}", "z-score")

# duration
histogram("norm_duration", numpy.linspace(-3,3,100), f"Speaker-Normalized Vowel Duration, {corpus}", "z-score")

# absolute f1
histogram("f1", numpy.linspace(200,1100,100), f"F1, {corpus}", "F1 (Hz)")

# absolute f2
histogram("f2", numpy.linspace(700,3000,100), f"F2, {corpus}", "F2 (Hz)")

# absolute duration
histogram("duration", numpy.linspace(0,0.3,30), f"Duration, {corpus}", "Duration (s)")


# f1 vs duration
scatter("norm_f1", "norm_duration", f"Speaker-Normalized F1 vs Vowel Duration, {corpus}", "F1 z-score", "Vowel Duration z-score")

# absolute f1 vs duration
scatter("f1", "duration", f"F1 vs Vowel Duration, {corpus}", "F1 (Hz)", "Duration (s)")
