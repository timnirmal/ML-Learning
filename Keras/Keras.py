# 1. load data
# 2. split data
# 3. normalize data
# 4. reshape data
# 5. make model
# 6. train model
# 7. save model
# 8. load model
# 9. predict

from random import randint

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Step 1. Making Dataset
# scrnario:
# 2100 people from 13-100 years old
# Half weer under 65 and half weer over 65
# 5% of child and 5% of adults, rest is middle ages

train_samples = []
train_labels = []

# Making the training set
for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)  # Have side effects

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)  # No side effects

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)  # No side effects

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)  # Have side effects

# Printing the training set
# for i in train_labels:
#    print(i)


# plot the training set
import matplotlib.pyplot as plt

plt.hist(train_samples, bins=20, range=(13, 100))
plt.xlabel('Age')
plt.ylabel('Number of people')
plt.title('Histogram of Age')
plt.show()

# Step 2. Converting the training set to numpy array and shuffle
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)

# Step 3. Scaling the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

print(scaled_train_samples)

# If you're using GPU, you can use this
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Step 4. Making the model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()

# Step 5. Compiling and Fitting Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, epochs=30, batch_size=10, shuffle=True,
          verbose=2)

# Step 6. Evaluating the model
"""
# Or we Can completely get rid of creating testing set
# Using validation_split on fit function
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, epochs=30, batch_size=10, shuffle=True,
          verbose=2)
"""

# create testing set
test_samples = []
test_labels = []

for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)  # Have side effects

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)  # No side effects

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)  # No side effects

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)  # Have side effects

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

# Step 7. Predicting

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

print("\n\nPredictions:")

# for i in predictions:
#    print(i)

# plot predictions
import matplotlib.pyplot as plt

plt.hist(predictions, bins=20, range=(0, 1))
plt.xlabel('Predictions')
plt.ylabel('Number of people')
plt.title('Histogram of Predictions')
# show legend
plt.legend(['Have side effects', 'No side effects'])

plt.show()

rounded_predictions = np.argmax(predictions, axis=-1)  # To get most probable class

print("\n\nRound predictions:")

# for i in rounded_predictions:
#    print(i)

# plot rounded predictions
plt.hist(rounded_predictions, bins=2, range=(0, 2))
plt.xlabel('Predictions')
plt.ylabel('Number of people')
plt.title('Histogram of predictions')
# show labels
# plt.xticks(np.arange(2), ('No side effects', 'Have side effects'))
# show legend
plt.legend(['No side effects', 'Have side effects'])
plt.show()

# NOTE: Since we use softmax of 2, we get [probability of no side effects, probability of have side effects] as
#   prediction for each sample


# Step 8. Confusion Matrix
# In here we have "Labeled" data, So we see the actual labels and the predicted labels in the confusion matrix

from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This Function prints and plot the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    classes: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see https://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('cool')  ##https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

cm_plot_labels = ['No side effects', 'Have side effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Normalized confusion matrix')
