import os

os.makedirs("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/train")
os.makedirs("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/test")

genres = 'blues classical country disco hiphop metal pop reggae rock'
genres = genres.split()
for g in genres:
  path_audio = os.path.join('/content/audio3sec',f'{g}')
  os.makedirs(path_audio)
  path_train = os.path.join("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/train",f'{g}')
  path_test = os.path.join("C:/Users/beni1/Desktop/Önlab/spectograms5s_v2_overlap4s/test",f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)