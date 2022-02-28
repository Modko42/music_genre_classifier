from pydub import AudioSegment
import os

filename = "/content2/watchtower.mp3"
song  =  os.path.join(filename)
newAudio = AudioSegment.from_wav(song)
t1 = 5000
t2 = 8000
new = newAudio[t1:t2]
new.export(f'/content2/custom_watchtower3s.wav', format="wav")
