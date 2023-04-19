import os

import pydub
import glob
import pandas as pd
import time

corpus_name = "TIMIT"
dir = "timit/TIMIT_ORIG/train"
SAMPLE_RATE = 16000

raw_files = [i.replace(".wav","") for i in glob.glob(dir + "/**/*.wav",recursive=True)]
raw_files = sorted(raw_files)
total = len(raw_files)
print(total)
print(raw_files[:20])

df = pd.DataFrame(columns=[
    "corpus",
    "corpus_path",
    "extracted_filename",
    "phone",
    "pred_phone",
    "next_phone",
    "start_time",
    "end_time",
    "duration",
    "word",
    "pred_word",
    "next_word",
    "is_word_onset",
    "speaker_id",
    "speaker_gender"
])

phone_dirs = "aa ah aw ax-h ay bcl d dh eh em eng er f gcl hv ix jh kcl m ng ow p pcl r sh tcl uh ux w z ae ao ax axr b ch dcl dx el en epi ey g hh ih iy k l n nx oy pau q s t th uw v y zh".split(" ")
for p in phone_dirs:
    try:
        os.makedirs(f"timit_out/{p}")
    except:
        pass
def get_word_of_phone(words, phone):
    s,e,p = phone
    for word in words:
        ws, we, w = word
        if ws <= s <= we:
            return w
    return None

def is_word_onset(words, phone):
    s,e,p = phone
    for word in words:
        ws, we, w = word
        if s == ws:
            return True
    return False

def get_rel_phone(phones, phone, code):
    s, e, p = phone

    for i in phones:
        ps, pe, pp = i
        if code == "prev" and pe == s:
            return pp
        elif code == "next" and e == ps:
            return pp
    return None

def get_rel_word(words, phone, code):
    s, e, p = phone

    found = None
    for word in words:
        ws, we, ww = word
        if ws <= s <= we:
            found = word
            break
    if found is None:
        return None
    cur_start, cur_end, cur_word = found
    for word in words:
        ws, we, ww = word
        if code == "prev" and we == cur_start:
            return ww
        elif code == "next" and cur_end == ws:
            return ww
    return None

def ftime(eta):
    eta_s = int(eta) % 60
    eta_m = int((eta // 60) % 60)
    eta_h = int((eta // (60 * 60)))
    return f"{eta_h}h{eta_m}m{eta_s}s"

count = 0
files_analyzed = 0
count_archived = 0

data_lists = []

a_start_time = time.time()

for f in raw_files:
    wav = f"{f}.wav"
    words = f"{f}.wrd"
    phone = f"{f}.phn"

    try:
        with open(words,"r") as wf:
            word_intervals = [i.split(" ") for i in wf.read().splitlines()]
            word_intervals = [[int(i[0]), int(i[1]), i[2]] for i in word_intervals]
    except:
        print(f"ERR: {f} HAS NO WORDS FILE, SKIPPING")
        continue

    with open(phone, "r") as pf:
        phones = pf.read().splitlines()
    phones = [i.split(" ") for i in phones]
    phones = [[int(i[0]), int(i[1]), i[2]] for i in phones]
    phones = [i for i in phones if i[2] != "h#"]

    #print(word_intervals)
    #print(phones)
    #print(f)

    dirs = f.split("/")

    speaker = dirs[-2]
    speaker_gender = speaker[0]
    dialect_region = dirs[-3]
    #print(speaker, dialect_region)

    audio = pydub.AudioSegment.from_wav(wav)

    for phone in phones:
        s, e, p = phone
        st = s / SAMPLE_RATE
        et = e / SAMPLE_RATE
        duration = et - st

        word = get_word_of_phone(word_intervals, phone)
        word_onset = is_word_onset(word_intervals, phone)

        prev_phone = get_rel_phone(phones, phone, "prev")
        next_phone = get_rel_phone(phones, phone, "next")
        prev_word = get_rel_word(word_intervals, phone, "prev")
        next_word = get_rel_word(word_intervals, phone, "next")

        output_dir = f"timit_out/{p}/"
        output = output_dir + f"{dialect_region}_{speaker}_{word}_{p}_{s}.wav".replace("'","")

        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        cropped = audio[st*1000:et*1000]
        cropped.export(output,format="wav")


        data_lists.append([
            corpus_name, # corpus
            f, # corpus path
            output, # extracted filename
            p, # phone
            prev_phone, # previous phoneme
            next_phone, # next phoneme
            st, # start time
            et, # end time
            duration, # duration
            word, # word
            prev_word, # previous word
            next_word, # next word
            word_onset, # is word onset?
            speaker, # speaker id
            speaker_gender # speaker gender
        ])

        count += 1
        if count % 100 == 0:
            print(f"{count} phones analyzed")

    files_analyzed += 1
    if files_analyzed % 30 == 0:
        cur_time = time.time()
        a_duration = cur_time - a_start_time
        eta = ((a_duration / files_analyzed) * total) - a_duration

        print(f"{files_analyzed}/{total} files analyzed (ETA {ftime(eta)}) (Last {f})")

print(f"{count} phonemes extracted. Took {ftime(time.time() - a_start_time)}")
df = pd.DataFrame(data_lists, columns=[
    "corpus",
    "corpus_path",
    "extracted_filename",
    "phone",
    "pred_phone",
    "next_phone",
    "start_time",
    "end_time",
    "duration",
    "word",
    "pred_word",
    "next_word",
    "is_word_onset",
    "speaker_id",
    "speaker_gender"
])
print(df)
df.to_pickle("timit.pkl")