import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

train_dir = 'data/pneumonia/chest_xray/train'
test_dir = 'data/pneumonia/chest_xray/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale = 1./255.,)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(224, 224))
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=False, batch_size=20, class_mode='binary', target_size=(224, 224))

mobilenet_model = MobileNetV3(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

for layer in mobilenet_model.layers[:-11]:
    layer.trainable = False

for layer in mobilenet_model.layers[-11:]:
    layer.trainable = True
    
model = Sequential()
model.add(mobilenet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

history = model.fit(train_generator, validation_data=test_generator, steps_per_epoch=len(train_generator), epochs=20, validation_steps=len(test_generator))

model.save("results/mobilenet/model.h5")

accuracy = history.history['accuracy']
val_accuracy  = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(dpi=200)

plt.subplot(2, 2, 1)
plt.plot(accuracy, label = "Training accuracy")
plt.plot(val_accuracy, label="Validation accuracy")
plt.ylim(0, 1)
plt.legend()
plt.title("Training vs validation accuracy")

plt.subplot(2, 2, 2)
plt.plot(loss, label = "Training loss")
plt.plot(val_loss, label="Validation loss")
plt.ylim(0, 0.5)
plt.legend()
plt.title("Training vs validation loss")

plt.show()

y_pred = model.predict(test_generator)
y_pred = 1 * (y_pred > 0.5)

print(confusion_matrix(test_generator.classes, y_pred))
print(classification_report(test_generator.classes, y_pred))