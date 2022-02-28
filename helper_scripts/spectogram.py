import numpy
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()


for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.path.join('/content/audio3sec_plswork', f"{g}")):
        song = os.path.join(f'/content/audio3sec_plswork/{g}', f'{filename}')
        j = j + 1
        print(filename)
        y, sr = librosa.load(song, duration=3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        #img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
        #img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
        #img = 255 - img
        #librosa.display.specshow(mels, cmap='gray_r', y_axis='linear')
        #plt.colorbar(format='%+2.0f dB')
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        #plt.imshow(img)
        plt.axis('off')
        plt.savefig(f'/content/spectrograms3sec/train/{g}/{g + str(j)}.png')
        plt.close()