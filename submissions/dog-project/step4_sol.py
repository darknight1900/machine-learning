from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from tqdm import tqdm

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
    
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


network = 'VGG16'
bottleneck_features = np.load('bottleneck_features/Dog{}Data.npz'.format(network))
train_feats = bottleneck_features['train']
valid_feats = bottleneck_features['valid']
test_feats = bottleneck_features['test']


model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_feats.shape[1:]))
model.add(Dense(133, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.{}.hdf5'.format(network), 
                               verbose=1, save_best_only=True)

model.fit(train_feats, train_targets, 
          validation_data=(valid_feats, valid_targets),
          epochs=20, batch_size=128, callbacks=[checkpointer], verbose=1)

VGG16_model.load_weights('saved_models/weights.best.{}.hdf5'.format(network))
model_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_feats]

# report test accuracy
test_accuracy = 100*np.sum(np.array(model_predictions)==np.argmax(test_targets, axis=1))/len(model_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
