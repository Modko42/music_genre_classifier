import os
import random
import shutil

genres = 'blues classical country disco hiphop metal pop reggae rock'
genres = genres.split()

directory = "C:/Users/beni1/Desktop/Önlab/spectograms6s_overlap3s/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  random.shuffle(filenames)
  test_files = filenames[0:100]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"C:/Users/beni1/Desktop/Önlab/spectograms6s_overlap3s/test/" + f"{g}")