import pandas as pd
import statistics

input_f = "BUC_vowels.csv"
output_f = "BUC_vowels_norm.csv"
grab_formants = False
speaker_field = "speaker"

db = pd.read_csv(input_f)  #"timit.pkl")

print(db)
'''
vowels = [
    "iy",
    "ih",
    "eh",
    "ey",
    "ae",
    "aa",
    "aw",
    "ay",
    "ah",
    "ao",
    "oy",
    "ow",
    "uh",
    "uw",
    "ux",
    "er",
    "ax",
    "ix",
    # "axr", --ignoring schwar?
    "ax-h"
]
'''
vowels = [
    "AE", "EH"
]
# drop non-vowels
db = db[db["phone"].isin(vowels)]

if grab_formants:
    # ignore full filename
    db["extracted_filename"] = db["extracted_filename"].map(lambda x: x.split("/")[-1])


    formants = pd.DataFrame()
    for p in vowels:
        f = pd.read_csv(f"formant_extr/formant_data/{p}.tsv",sep="\t",names=[
            "extracted_filename", "interval", "label", "midPoint", "f1", "f2", "numFormants", "maxFormant"
        ])
        f = f.iloc[1: , :]

        formants = pd.concat([formants, f[["extracted_filename", "f1", "f2"]]], axis=0)

    print(formants)
    db = pd.merge(db, formants, on="extracted_filename", how="left")

for i in ["f1", "f2", "duration"]:
    db[i] = db[i].map(float)

db["deltaFormant"] = db["f2"] - db["f1"]

print(db)
print(db.columns)

# Make the db from whence we get stats


def get_normalized_dictionary(column):
    stats = {}
    for speaker in db[speaker_field].unique():
        # get all rows where the "speaker" column equals the current speaker
        rows = db.loc[db[speaker_field] == speaker, :]

        # calculate the mean and standard deviation of the "f1" values for these rows
        vals = rows[column].dropna()
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)

        # add the mean and standard deviation to the dictionary, keyed by the speaker value
        stats[speaker] = (mean, std)
    return stats

print(get_normalized_dictionary("f1"))
def generate_z_scores(input_column):
    statsdict = get_normalized_dictionary(input_column)
    print(len(statsdict))
    def z_score(n, s):
        mean, stdev = statsdict[s]
        return (n - mean)/stdev

    output_column = "norm_" + input_column
    db[output_column] = db.apply(lambda row: z_score(row[input_column], row[speaker_field]), axis=1)

generate_z_scores("f1")
generate_z_scores("f2")
generate_z_scores("deltaFormant")
generate_z_scores("duration")


print(db)

db.to_csv(output_f)

print(get_normalized_dictionary("duration"))
