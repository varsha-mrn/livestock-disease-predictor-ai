# train_audio_model.py - simple audio training script using MFCCs
import os, numpy as np
try:
    import librosa
except Exception:
    print("librosa not installed. pip install librosa soundfile")
    raise
from tensorflow.keras import layers, models, optimizers
AUDIO_DIR = "cow_audio"
def extract(path, sr=22050, dur=3.0):
    y, _ = librosa.load(path, sr=sr, duration=dur)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)
X, y = [], []
classes = []
if not os.path.exists(AUDIO_DIR):
    print("No cow_audio/ folder found. Add audio under cow_audio/<class>/")
else:
    for i,cls in enumerate(sorted(os.listdir(AUDIO_DIR))):
        classes.append(cls)
        folder = os.path.join(AUDIO_DIR, cls)
        for f in os.listdir(folder):
            if f.lower().endswith(('.wav','.mp3')):
                X.append(extract(os.path.join(folder,f)))
                y.append(i)
    X = np.array(X)
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=len(classes))
    model = models.Sequential([layers.Input(shape=(20,)), layers.Dense(64,activation='relu'), layers.Dense(32,activation='relu'), layers.Dense(len(classes), activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X,y, epochs=10, batch_size=8)
    model.save(os.path.join('models','audio_model_trained.h5'))
    print('Saved trained audio model to models/audio_model_trained.h5')