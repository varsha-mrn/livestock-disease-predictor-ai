import os, math, random, tempfile
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Optional TensorFlow; app will run in demo mode if not installed.
TF_AVAILABLE = True
try:
    import numpy as np
    from PIL import Image
    from tensorflow.keras import models, layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    TF_AVAILABLE = False
    import numpy as np
    from PIL import Image

# -------------------------
# Symptom dataset + Naive Bayes
# -------------------------
SYMPTOMS = [
    "High fever","Coughing","Nasal discharge","Diarrhea",
    "Lethargy","Loss of appetite","Lameness","Swollen joints",
    "Skin lesions","Rapid breathing","Drooling"
]
DISEASES = [
    "Foot-and-mouth disease","Pneumonia","Bovine viral diarrhea (BVD)",
    "Mastitis","Parasitic infection","Healthy"
]

def generate_symptom_dataset(n=80):
    data = []
    for _ in range(n):
        disease = random.choice(DISEASES)
        symptoms = random.sample(SYMPTOMS, random.randint(2,5))
        if disease=="Foot-and-mouth disease":
            symptoms += ["Drooling","Skin lesions"]
        elif disease=="Pneumonia":
            symptoms += ["Coughing","Nasal discharge","Rapid breathing"]
        elif disease=="Bovine viral diarrhea (BVD)":
            symptoms += ["High fever","Diarrhea"]
        elif disease=="Mastitis":
            symptoms += ["Loss of appetite","Lethargy"]
        elif disease=="Parasitic infection":
            symptoms += ["Lethargy","Swollen joints"]
        data.append((list(set(symptoms)), disease))
    return data

TRAINING_DATA = generate_symptom_dataset()

class NaiveBayesClassifier:
    def __init__(self, symptoms, diseases, data):
        self.symptoms = symptoms
        self.diseases = diseases
        self.train(data)
    def train(self, data):
        self.prior = defaultdict(float)
        self.cond_prob = defaultdict(lambda: defaultdict(float))
        disease_counts = defaultdict(int)
        for symptoms, disease in data:
            disease_counts[disease]+=1
        total_samples = len(data)
        for disease in self.diseases:
            self.prior[disease]=(disease_counts[disease]+1)/(total_samples+len(self.diseases))
        for disease in self.diseases:
            for sym in self.symptoms:
                count_present = sum(1 for s,d in data if d==disease and sym in s)
                count_disease = disease_counts[disease]
                self.cond_prob[disease][sym]=(count_present+1)/(count_disease+2)
    def predict(self, present_symptoms):
        log_scores={}
        for disease in self.diseases:
            log_prob=math.log(self.prior[disease])
            for sym in self.symptoms:
                p=self.cond_prob[disease][sym]
                log_prob += math.log(p if sym in present_symptoms else (1-p))
            log_scores[disease]=log_prob
        max_log=max(log_scores.values())
        exp_scores={d:math.exp(v-max_log) for d,v in log_scores.items()}
        total=sum(exp_scores.values())
        probs={d:v/total for d,v in exp_scores.items()}
        best=max(probs,key=probs.get)
        return best, probs[best], probs

symptom_classifier = NaiveBayesClassifier(SYMPTOMS, DISEASES, TRAINING_DATA)

# -------------------------
# Image model (demo / transfer learning if TF available)
# -------------------------
class ImageModel:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.classes = ["Healthy", "Foot-and-mouth disease", "Pneumonia"]
        self.dataset_dir = "cow_images"
        if TF_AVAILABLE:
            self.prepare_dataset()
            self.train_or_load()
        else:
            print("TensorFlow not available – image model disabled (demo mode).")

    def prepare_dataset(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            for cls in self.classes:
                os.makedirs(os.path.join(self.dataset_dir, cls), exist_ok=True)
            colors = {"Healthy": "green", "Foot-and-mouth disease": "red", "Pneumonia": "blue"}
            for cls, color in colors.items():
                for i in range(20):
                    img = Image.new("RGB", (128, 128), color)
                    img.save(os.path.join(self.dataset_dir, cls, f"{cls}_{i}.png"))

    def train_or_load(self):
        model_path = os.path.join("models","image_model_dummy.h5")
        if os.path.exists(model_path) and TF_AVAILABLE:
            try:
                self.model = models.load_model(model_path)
                return
            except Exception:
                self.model = None

        if not TF_AVAILABLE:
            return

        try:
            datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                validation_split=0.2
            )
            train_gen = datagen.flow_from_directory(
                self.dataset_dir,
                target_size=self.img_size,
                class_mode="categorical",
                subset="training"
            )
            val_gen = datagen.flow_from_directory(
                self.dataset_dir,
                target_size=self.img_size,
                class_mode="categorical",
                subset="validation"
            )
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
            base_model.trainable = False
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dense(128, activation='relu')(x)
            output = layers.Dense(len(self.classes), activation='softmax')(x)
            self.model = models.Model(inputs=base_model.input, outputs=output)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(train_gen, epochs=3, validation_data=val_gen, verbose=1)
            self.model.save(model_path)
        except Exception as e:
            print("Image training failed or not possible:", e)
            self.model = None

    def predict_image(self, img_path):
        if not TF_AVAILABLE or self.model is None:
            img = Image.open(img_path).resize(self.img_size)
            arr = np.array(img).astype(np.float32) / 255.0
            mean_color = arr.mean(axis=(0,1))
            r,g,b = mean_color
            if r > 0.5 and r > (g + 0.15):
                return "Foot-and-mouth disease", 0.75
            if b > 0.45 and b > (r + 0.05):
                return "Pneumonia", 0.60
            return "Healthy", 0.85

        img = load_img(img_path, target_size=self.img_size)
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0)
        preds = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        return self.classes[idx], float(preds[idx])

image_model = ImageModel()

# -------------------------
# Audio model (MFCC-based small NN)
# -------------------------
class AudioModel:
    def __init__(self):
        self.model = None
        self.sample_rate = 22050
        self.audio_dir = "cow_audio"
        self.classes = ["Healthy", "Pneumonia", "Foot-and-mouth disease"]
        if TF_AVAILABLE:
            self.prepare_dataset()
            self.train_or_load()
        else:
            print("TensorFlow not available – audio model disabled (demo mode).")

    def extract_features(self, file_path):
        try:
            import librosa
        except Exception:
            raise RuntimeError("librosa not installed")
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    def prepare_dataset(self):
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
            for cls in self.classes:
                os.makedirs(os.path.join(self.audio_dir, cls), exist_ok=True)
            print("Add your audio samples (.wav) inside cow_audio/<class_name>/ to train a real model.")

    def train_or_load(self):
        model_path = os.path.join("models","audio_model_dummy.h5")
        if os.path.exists(model_path) and TF_AVAILABLE:
            try:
                self.model = models.load_model(model_path)
                return
            except Exception:
                self.model = None

        if not TF_AVAILABLE:
            return

        # build tiny dataset from files if present
        X, y = [], []
        for idx, cls in enumerate(self.classes):
            folder = os.path.join(self.audio_dir, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.wav','.mp3')):
                    fpath = os.path.join(folder, fname)
                    try:
                        feat = self.extract_features(fpath)
                        X.append(feat)
                        y.append(idx)
                    except Exception:
                        pass
        if not X:
            import numpy as _np
            X = _np.random.randn(60, 20)
            y = _np.random.randint(0, len(self.classes), 60)

        X = np.array(X)
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(y, num_classes=len(self.classes))

        self.model = models.Sequential([
            layers.Input(shape=(20,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        self.model.save(model_path)

    def predict_audio(self, file_path):
        if not TF_AVAILABLE or self.model is None:
            # simple demo heuristic: file size based pseudo-prediction
            try:
                size = os.path.getsize(file_path)
                if size % 3 == 0:
                    return "Pneumonia", 0.65
                if size % 2 == 0:
                    return "Foot-and-mouth disease", 0.60
            except Exception:
                pass
            return "Healthy", 0.80
        feat = self.extract_features(file_path)
        arr = np.expand_dims(feat, axis=0)
        preds = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        return self.classes[idx], float(preds[idx])

audio_model = AudioModel()

# -------------------------
# Flask app configuration and routes
# -------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_IMG = {"png","jpg","jpeg","bmp"}
ALLOWED_AUDIO = {"wav","mp3","ogg","flac","m4a"}

def allowed_file(filename, extset):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in extset

@app.route('/')
def index():
    return render_template('index.html', symptoms=SYMPTOMS)

@app.route('/predict-symptoms', methods=['POST'])
def predict_symptoms():
    picked = request.form.getlist('symptom')
    if not picked:
        return jsonify({'error':'no symptom selected'}), 400
    disease, conf, probs = symptom_classifier.predict(picked)
    probs_sorted = sorted([(d, float(p)) for d,p in probs.items()], key=lambda x:-x[1])
    return jsonify({'selected': picked, 'predicted': disease, 'confidence': float(conf), 'probs': probs_sorted})

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error':'no file'}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error':'empty filename'}), 400
    if not allowed_file(f.filename, ALLOWED_IMG):
        return jsonify({'error':'file type not allowed'}), 400
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    label, conf = image_model.predict_image(path)
    return jsonify({'predicted': label, 'confidence': float(conf)})

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error':'no file'}), 400
    f = request.files['audio']
    if f.filename == '':
        return jsonify({'error':'empty filename'}), 400
    if not allowed_file(f.filename, ALLOWED_AUDIO):
        return jsonify({'error':'file type not allowed'}), 400
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    label, conf = audio_model.predict_audio(path)
    return jsonify({'predicted': label, 'confidence': float(conf)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
