import parselmouth
import pydub
import json
import glob

s_duration = 0.255
t_duration = 0.430

def get_data(filename):
    sound = parselmouth.Sound(filename)
    formant_burg = sound.to_formant_burg()

    audio = pydub.AudioSegment.from_wav(filename)
    full_duration = len(audio)/1000
    vowel_duration = full_duration - s_duration - t_duration
    formant_point = s_duration + (vowel_duration / 2)
    f1 = parselmouth.Formant.get_value_at_time(formant_burg,1,formant_point)
    f2 = parselmouth.Formant.get_value_at_time(formant_burg,2,formant_point)
    return (f1, f2, vowel_duration)


directory = "setsat"

results = {}
for f in glob.glob(f"{directory}/*.wav"):
    results[f] = get_data(f)

with open("set_sat_results.json", "w+") as f:
    json.dump(results, f)




