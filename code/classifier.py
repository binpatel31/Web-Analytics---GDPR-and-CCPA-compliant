import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import defaultdict
import numpy.random as rng
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Lambda, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from tensorflow import keras
import pickle

class Classifier:

    def __init__(self):

        with open('vendor_tokenizer.pickle', 'rb') as handle:
            self.vendor_tokenizer = pickle.load(handle)

        with open('language_tokenizer.pickle', 'rb') as handle:
            self.language_tokenizer = pickle.load(handle)

        with open('useragent_tokenizer.pickle', 'rb') as handle:
            self.useragent_tokenizer = pickle.load(handle)

        with open('browser_tokenizer.pickle', 'rb') as handle:
            self.browser_tokenizer = pickle.load(handle)

        with open('platform_tokenizer.pickle', 'rb') as handle:
            self.platform_tokenizer = pickle.load(handle)

        with open('os_tokenizer.pickle', 'rb') as handle:
            self.os_tokenizer = pickle.load(handle)

        with open('scaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open('loader.pickle', 'rb') as handle:
            self.loader = pickle.load(handle)



    def get_model(self):
        input_shape = (72)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        model = Sequential()
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))

        encoded_l = model(left_input)
        encoded_r = model(right_input)
        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid')(L1_distance)
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        siamese_net.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
        siamese_net.load_weights("model.h5")

        self.model = siamese_net

        return siamese_net

    def dump_pickles(self):

        with open('vendor_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.vendor_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('language_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.language_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('useragent_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.useragent_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('browser_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.browser_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('platform_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.platform_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('os_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.os_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('scaler.pickle', 'wb') as handle:
            pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('loader.pickle', 'wb') as handle:
            pickle.dump(self.loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save_weights("model.h5")


    def test(self, platform, os, browser, timezone, resWidth, resHeight, resChannel, useragent, vendor, language):
        platform = self.platform_tokenizer.texts_to_sequences([platform])
        platform = pad_sequences(platform, padding="post", maxlen=3)
        os = self.os_tokenizer.texts_to_sequences(os)
        os = pad_sequences(os, padding="post", maxlen=3)
        browser = self.browser_tokenizer.texts_to_sequences([browser])
        useragent = self.useragent_tokenizer.texts_to_sequences([useragent])
        useragent = pad_sequences(useragent, padding="post", maxlen=51)
        vendor = self.vendor_tokenizer.texts_to_sequences([vendor])
        vendor = pad_sequences(vendor, padding="post", maxlen=5)
        language = self.language_tokenizer.texts_to_sequences([language])
        language = pad_sequences(language, padding="post", maxlen=5)

        test = []
        test.extend(platform[0])
        test.extend(os[0])
        test.extend(browser[0])
        test.extend([timezone])
        test.extend([resWidth])
        test.extend([resHeight])
        test.extend([resChannel])
        test.extend(useragent[0])
        test.extend(vendor[0])
        test.extend(language[0])

        test = np.array(test).reshape(1,-1)
        test = self.scaler.transform(test)

        siamese_net = self.get_model()

        test_image = np.array(list(self.loader.data['train'][0])).reshape(1, len(self.loader.data['train'][0]))
        cut_off = siamese_net.predict([test_image, test_image])[0, 0]

        label, val, _, _ = self.loader.test(test, siamese_net, cut_off)
        if val > 0.01:
            label = self.loader.train_single(test[0], siamese_net)
            self.dump_pickles()
            print("New User")
            return 1, label

        return -1, label


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, X, Y, data_subsets=["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}

        self.data["train"] = X
        self.data["val"] = X
        self.categories["train"] = Y
        self.categories["val"] = Y

        self.data_idx = defaultdict(list)
        for x, y in zip(X, Y):
            self.data_idx[y].append(x)

    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_examples, features = X.shape
        n_classes = len(set(self.categories[s]))

        # randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, features)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]
            #             print(len(self.data_idx[category]), category)
            idx_1 = rng.randint(0, len(self.data_idx[category]))
            pairs[0][i, :] = self.data_idx[category][idx_1].reshape(features)
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
            idx_2 = rng.randint(0, len(self.data_idx[category_2]))
            pairs[1][i, :] = self.data_idx[category_2][idx_2].reshape(features)
        return pairs, targets

    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)

    def make_oneshot_task(self, N, s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X = self.data[s]
        n_examples, features = X.shape
        n_classes = len(set(self.categories[s]))

        categories = rng.choice(range(n_classes), size=(N,), replace=False)
        true_category = categories[0]
        ex1 = rng.choice(len(self.data_idx[true_category]), replace=False, size=(1))[0]

        test_image = np.array(list(self.data_idx[true_category][ex1]) * N).reshape(N, features)

        support_set = []
        for i in range(N):
            idx = rng.choice(len(self.data_idx[categories[i]]), replace=False, size=(1))[0]
            support_set.append(self.data_idx[categories[i]][idx])

        support_set = np.array(support_set)
        support_set[0] = test_image[0]
        support_set = support_set.reshape(N, features)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def test(self, X, model, p, s="train"):
        categories = [i for i in range(len(set(self.categories[s])))]
        N = len(categories)
        n_examples, features = self.data[s].shape
        test_image = np.array(list(X)).reshape(1, features)

        ans = -1
        max_ans = -1
        lowest_sim = float("inf")
        max_prob = 0
        for i in range(N):
            idx = rng.choice(len(self.data_idx[categories[i]]), replace=False, size=(1))[0]
            support_image = np.array(self.data_idx[categories[i]][idx]).reshape(1, features)
            inputs = [test_image, support_image]
            probs = model.predict(inputs)
            if abs(probs[0, 0] - p) < lowest_sim:
                lowest_sim = abs(probs[0, 0] - p)
                ans = i
            if probs[0, 0] > max_prob:
                max_prob = probs[0, 0]
                max_ans = i

        #         if lowest_sim>0.01:
        #             ans = -1

        return ans, lowest_sim, max_ans, max_prob

    def train_single(self, x, model, label=-1, N=-1):
        categories = [i for i in range(len(set(self.categories['train'])))]
        last_label = len(categories)
        if N == -1:
            N = len(categories)
        n_examples, features = self.data['train'].shape
        train_image = np.array(list(x)).reshape(1, features)

        avg_loss = 0

        for i in range(N):
            idx = rng.choice(len(self.data_idx[categories[i]]), replace=False, size=(1))[0]
            support_image = np.array(self.data_idx[categories[i]][idx]).reshape(1, features)
            inputs = [train_image, support_image]
            if categories[i] == label:
                targets = [1, 1]
            else:
                targets = [1, 0]
            avg_loss += model.train_on_batch(inputs)

        if label != -1:
            self.data_idx[label] = np.append(self.data_idx[label], [x], axis=0)
            self.data['train'] = np.append(self.data['train'], [x], axis=0)
            self.categories['train'] = np.append(self.categories['train'], [label], axis=0)
        else:
            self.data_idx[last_label] = [x]
            self.data['train'] = np.append(self.data['train'], [x], axis=0)
            self.categories['train'] = np.append(self.categories['train'], [last_label], axis=0)

        return last_label

    def test_oneshot(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
        return percent_correct

    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),

                            )

