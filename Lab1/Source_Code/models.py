from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3), num_classes=10, learning_rate=1e-3, dropout=0.5):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
