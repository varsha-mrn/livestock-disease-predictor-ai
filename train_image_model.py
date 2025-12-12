# train_image_model.py - simple transfer-learning training script
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks, optimizers
IMG_SIZE = (128,128)
BATCH_SIZE = 16
DATA_DIR = "cow_images"  # expects train/val subfolders or class subfolders
if not os.path.exists(DATA_DIR):
    print("No cow_images/ folder found. Add images in cow_images/<class>/ and rerun.")
else:
    ds = image_dataset_from_directory(DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')
    num_classes = ds.element_spec[1].shape[-1]
    base = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base.input, out)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds, epochs=5)
    model.save(os.path.join('models','image_model_trained.h5'))
    print('Saved trained image model to models/image_model_trained.h5')