import pandas
from os.path import exists
import os
import pydub
import sys
import numpy as np
import parselmouth

corpus = "WSJ"
if len(sys.argv) != 1:
    corpus = sys.argv[1]

metadirectory = f"/fs/clip-realspeech/corpora/spock-format/{corpus}/aligned_data/"
alignment_location = metadirectory + "alignment.txt"
wav_location = metadirectory + "wavs/"
output_directory = f"output/{corpus}"
speaker_gender = metadirectory + f"spk2gender_{corpus}.txt"
file_speaker = metadirectory + "utt2spk.txt"

desired_phones = ["AE", "EH"]
for p in desired_phones:
    try:
        os.makedirs(f"{output_directory}/{p}")
    except FileExistsError:
        pass

temp = "output/unquoted_alignment.csv"
if not exists("WSJ_phonedb.csv") or True:
    print("Expunging Quotes")
    with open(alignment_location, "r") as f:
        with open(temp, "w+") as g:
            g.write(f.read().replace('"',''))
    print("Reading Database from alignment.txt")
    names = ["filename","start_time","end_time","dont_know","phone","word"]
    data = pandas.read_csv(temp,sep=" ",names=names)
    print("Adding word onset flag")
    data["is_word_onset"] = data["word"].notnull()
    print("Adding speaker")
    speakers = pandas.read_csv(file_speaker,sep=" ",names=["filename","speaker"])
    data = pandas.merge(data, speakers, on="filename", how="inner")
    print("Adding gender")
    gender = pandas.read_csv(speaker_gender,sep=" ",names=["speaker","gender"])
    data = pandas.merge(data, gender, on="speaker", how="inner")
    data["filepath"] = wav_location + data['filename'] + ".wav"
    data.to_csv(f"{corpus}_phonedb.csv")
else:
    print("Using Existing Database")
    data = pandas.read_csv(f"{corpus}_phonedb.csv")

print(f"Database made, {len(data)} entries.")

filtered = data.loc[data["phone"].isin(desired_phones)]
filtered["f1"] = ''
filtered["f2"] = ''
filtered["duration"] = ''


for wavname in filtered.filename.unique():
    print(f"Extracting from {wavname}")
    in_this_file = filtered.loc[filtered["filename"] == wavname]
    # in_this_file.index = list(range(len(in_this_file)))

    try:
        sound = parselmouth.Sound(f"{wav_location}/{wavname}.wav")
        formant_burg = sound.to_formant_burg()
    except:
        print("ERROR: FAILED TO MAKE PARSELMOUTH SOUND OBJECT FOR " + f"{wav_location}/{wavname}.wav")
        continue
    for n, row in in_this_file.iterrows():
        s = row["start_time"]
        e = row["end_time"]
        median = (s + e)/2
        filtered.at[n, "f1"] = parselmouth.Formant.get_value_at_time(formant_burg,1,median)
        filtered.at[n, "f2"] = parselmouth.Formant.get_value_at_time(formant_burg,2,median)
        filtered.at[n, "duration"] = e - s

        # phone_audio = audio[s:e]
        # phone_audio.export(f"{output_directory}/{p}/{wavname}_{n}.wav", format="wav")

filtered.to_csv(f"{corpus}_vowels.csv")
