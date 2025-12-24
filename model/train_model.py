from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from rcnn_model import build_rcnn_model

# =========================
# Dataset paths
# =========================
train_dir = r"C:/Users/Dell/Desktop/Railway Code/Dataset/Train"
valid_dir = r"C:/Users/Dell/Desktop/Railway Code/Dataset/Validiation"

# =========================
# Image Augmentation (IMPROVED)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# =========================
# Data loaders
# =========================
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# =========================
# Build model
# =========================
model = build_rcnn_model()

# =========================
# Compile model
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Callbacks (VERY IMPORTANT)
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "railway_defect_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# =========================
# Train model
# =========================
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=30,
)

model.save("track_model.h5")
print("✅ Best model saved as track_model.h5")
