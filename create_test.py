import os
import random
import shutil

genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()

directory = "C:/Users/beni1/Desktop/spectograms3s_v5/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  random.shuffle(filenames)
  test_files = filenames[0:100]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"C:/Users/beni1/Desktop/spectograms3s_v5/test/" + f"{g}")