
import numpy as np
import scipy
from matplotlib_inline.backend_inline import FigureCanvas
from scipy import misc
import glob
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import shutil
import random

os.makedirs('/content2/spectrograms3sec')
os.makedirs('/content2/spectrograms3sec/train')
os.makedirs('/content2/spectrograms3sec/test')

genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()
for g in genres:
  path_audio = os.path.join('/content2/audio3sec',f'{g}')
  os.makedirs(path_audio)
  path_train = os.path.join('/content2/gdrive/My Drive/spectrograms3sec/train',f'{g}')
  path_test = os.path.join('/content2/gdrive/My Drive/spectrograms3sec/test',f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)


from pydub import AudioSegment
i = 0
for g in genres:
  j=0
  print(f"{g}")
  for filename in os.listdir(os.path.join('/genres_original',f"{g}")):

    song  =  os.path.join(f'/genres_original/{g}',f'{filename}')
    j = j+1
    for w in range(0,10):
      i = i+1
      #print(i)
      t1 = 3*(w)*1000
      t2 = 3*(w+1)*1000
      newAudio = AudioSegment.from_wav(song)
      new = newAudio[t1:t2]
      new.export(f'/content2/audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")

for g in genres:
  j = 0
  print(g)
  for filename in os.listdir(os.path.join('/content2/audio3sec', f"{g}")):
    song = os.path.join(f'/content2/audio3sec/{g}', f'{filename}')
    j = j + 1
    print(filename)
    y, sr = librosa.load(song, duration=3)
    # print(sr)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.axis('off')
    plt.savefig(f'/content2/gdrive/My Drive/spectrograms3sec/train/{g}/{g + str(j)}.png')
    plt.close()