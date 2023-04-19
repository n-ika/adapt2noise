# Python Dependencies
These scripts were run with the following library versions. Other versions *might* work, but your mileage may vary.
```
pydub==0.25.1
numpy==1.24.1
pandas==1.5.3
praat-parselmouth==0.4.3
matplotlib==3.7.0
```
# Scripts
Scripts used can be found in `scripts/`. The following is a list of such scripts:

`phonegrabber_v3.py`
This script is used to extract data from a corpus in the `spock` format as on UMD's clusters (see below for description of format), and produce a CSV containing the relevant data. 
The relevant configuration for filepaths etc is at the top of the file, you will need to change the directory into which it points.

`phonegrabber_BUC.py` is identical to the above, accounting for an artifact of the BUC data on the UMD clusters being in a slightly different, less processed format. It is included for completeness, but you should not need to use it.

`normalization.py` takes a csv output by one of the other scripts and computes normalized per-speaker data, and produces another csv containing this data. The first few lines of the script control which files are input and output.

`vowel_norm_plots.py` takes the output of `normalization.py` and uses matplotlib to produce figures of the data. As before, paths can be configured in the script.

`timit_db.py` splits the entire TIMIT corpus into individual wav files formatted by phoneme, and `timit_vowelduration.py` strips the output database with only the relevant vowels. If you wish to analyze this database with `normalization.py`, 

`set_sat.py` was used to construct the database for the SET and SAT test data.

# Data
Data files used can be found in `data/`. CSVs with normalized results are suffixed with `_norm`. `TIMIT_vowels_norm_full.csv` uniquely was normalized over *all* vowels of the TIMIT corpus. `TIMIT_vowels_norm.csv` contains only the two vowels used in the paper (and is the data used in the model). 
# Description of Corpus Format
The following are necessary:
-- A file (defaulted to `segments.txt`) containing a space-delimited list of segment IDs, followed by the wav file that segment is in and its start and end times.
Example:
```
001c0201 001.wav 0.0 8.27125
001c0202 001.wav 8.77125 14.38325
001c0203 001.wav 14.88325 21.4145625
001c0204 001.wav 21.9145625 28.60075
001c0205 001.wav 29.10075 34.4673125
001c0206 001.wav 34.9673125 43.4208125
001c0207 001.wav 43.9208125 46.8276875
001c0208 001.wav 47.3276875 52.72375
001c0209 001.wav 53.22375 57.32725
001c020a 001.wav 57.82725 62.853
001c020b 001.wav 63.353 66.387875
<many many more lines>
```
-- A file (defaulted to `alignment.txt`) containing alignment data, with each line being a space-separated list containing the following:
`segment_id start_time end_time UNUSED phoneme word*` (`word*` is only included if the phone starts a word)
Example:
```
00fc0712 0.0 0.1175 1.0 SIL
00fc0712 0.1175 0.1875 1.0 IH IN
00fc0712 0.1875 0.2375 0.99999996 N
00fc0712 0.2375 0.3075 0.999999942857 P PUBLIC
00fc0712 0.3075 0.4175 1.0 AH
00fc0712 0.4175 0.4675 1.0 B
00fc0712 0.4675 0.5175 1.0 L
00fc0712 0.5175 0.5675 1.0 IH
00fc0712 0.5675 0.6375 0.999748857143 K
00fc0712 0.6375 0.7075 1.0 M MEMBERS
00fc0712 0.7075 0.7775 1.0 EH
00fc0712 0.7775 0.8275 1.0 M
00fc0712 0.8275 0.8675 1.0 B
00fc0712 0.8675 0.9275 1.0 ER
00fc0712 0.9275 1.0175 0.999999766667 Z
```
A file (defaulted to `utt_speaker.txt`) mapping segment IDs to speaker IDs
Example:
```
001c0201 001
001c0202 001
001c0203 001
001c0204 001
001c0205 001
001c0206 001
001c0207 001
001c0208 001
001c0209 001
001c020a 001
001c020b 001
001c020c 001
```
A file (defaulted to `spk2gender.txt`) mapping speaker IDs to their gender
Example:
```
001 m
002 f
00a f
00b m
00c m
00d m
00f f
010 m
011 f
012 m
013 m
014 f
```
A directory `wavs` containing one `wav` file per segment, named with the segment's ID.
