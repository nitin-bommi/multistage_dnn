from sklearn import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = 'data/pneumonia/chest_xray/test'
t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

test_datagen = ImageDataGenerator(rescale = 1./255.,)
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=False, batch_size=20, class_mode='binary', target_size=(224, 224))

inception = load_model('results/inception/model.h5')
resnet = load_model('results/resnet/model.h5')
densenet = load_model('results/densenet/model.h5')
mobilenet = load_model('results/mobilenet/model.h5')

inception_pred = inception.predict(test_generator)
resnet_pred = resnet.predict(test_generator)
densenet_pred = resnet.predict(test_generator)
mobilenet_pred = mobilenet.predict(test_generator)

for i in t:
    mob = 1 * (mobilenet_pred > i)
    den = 1 * (densenet_pred > i)
    inc = 1 * (inception_pred > i)
    res = 1 * (resnet_pred > i)

    final = mob * den * inc * res

    print(i)
    print(metrics.confusion_matrix(test_generator.classes, final))
    print('######')
    
for i in t:
    mob = 1 * (mobilenet_pred > i)
    den = 1 * (densenet_pred > i)
    inc = 1 * (inception_pred > i)
    res = 1 * (resnet_pred > i)

    final = 1 * ((mob + den + inc + res)) > 1

    print(i)
    print(metrics.confusion_matrix(test_generator.classes, final))
    print('######')