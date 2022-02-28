import os

from pydub import AudioSegment
i = 0

song = "C:/Users/beni1/Desktop/own_music/daggers/full.wav"

for w in range(0,40):
  i = i+1
  t1 = 3*(w)*1000
  t2 = 3*(w+1)*1000
  newAudio = AudioSegment.from_wav(song)
  new = newAudio[t1:t2]
  new.export(f'C:/Users/beni1/Desktop/own_music/daggers/3sec/spec{str(w)}.wav', format="wav")
