import os
import random
import shutil

genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()

directory = "/content2/spectrograms3sec/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  random.shuffle(filenames)
  test_files = filenames[0:100]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"/content2/spectrograms3sec/test/" + f"{g}")