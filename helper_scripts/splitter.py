import os

genres = 'blues classical country disco hiphop metal pop reggae rock'
genres = genres.split()


from pydub import AudioSegment
i = 0
for g in genres:
  j=0
  print(f"{g}")
  for filename in os.listdir(os.path.join('C:/Users/beni1/Desktop/Önlab/original_wavs', f"{g}")):

    song = os.path.join(f'/Users/beni1/Desktop/Önlab/original_wavs/{g}', f'{filename}')
    j = j+1
    for w in range(0,9):
      i = i+1
      #print(i)
      t1 = 3*(w)*1000
      t2 = t1+6000
      #print([t1,t2])
      newAudio = AudioSegment.from_wav(song)
      new = newAudio[t1:t2]
      new.export(f'C:/Users/beni1/Desktop/Önlab/audio6sec_overlap3s/{g}/{g+str(j)+str(w)}.wav', format="wav")