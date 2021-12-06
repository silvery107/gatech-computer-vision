import cv2
import tqdm
import pickle
import numpy as np
import cyvlfeat as vlfeat
from sklearn.svm import LinearSVC
from utils import load_image, load_image_gray
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.model_selection import cross_val_score

DTYPE = np.float32


def get_tiny_images(image_paths):
    # dummy feats variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for img_path in image_paths:
        img = load_image_gray(img_path).astype(np.float32)
        feat = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
        feat = feat.flatten()
        feat -= np.mean(feat, dtype=DTYPE)
        feat /= np.linalg.norm(feat)
        feats.append(feat)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def build_vocabulary(image_paths, vocab_size):
    # length of the SIFT descriptors that you are going to compute.
    dim = 128
    vocab = np.zeros((vocab_size, dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    feats = []
    for i in tqdm.trange(len(image_paths), desc='getting a vocab SIFT'):
        img = load_image_gray(image_paths[i]).astype(np.float32)
        _, descriptors = vlfeat.sift.dsift(img, step=16, fast=True, float_descriptors=True)
        d_norm = np.linalg.norm(descriptors, axis=1)
        idx_nonzero = np.nonzero(d_norm)
        d_norm = d_norm[idx_nonzero]
        descriptors = descriptors[idx_nonzero].astype(DTYPE)
        d_norm = np.linalg.norm(descriptors, axis=1)
        descriptors /= d_norm[:, None]
        feats.append(descriptors)

    feats = np.vstack([feat for feat in feats])
    vocab = vlfeat.kmeans.kmeans(
        np.asarray(feats, dtype=DTYPE),
        vocab_size,
        initialization='PLUSPLUS',  # RANDSEL, PLUSPLUS
        distance='l2',  # l1, l2
        algorithm='LLOYD')  # LLOYD, ELKAN

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab


def get_bags_of_sifts(image_paths, vocab_filename):
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for i in tqdm.trange(len(image_paths), desc='getting bags of SIFT'):
        img = load_image_gray(image_paths[i]).astype(DTYPE)
        _, descriptors = vlfeat.sift.dsift(img, step=4, fast=True, float_descriptors=True)
        d_norm = np.linalg.norm(descriptors, axis=1)
        idx_nonzero = np.nonzero(d_norm)
        d_norm = d_norm[idx_nonzero].astype(DTYPE)
        descriptors = descriptors[idx_nonzero].astype(DTYPE)
        descriptors /= d_norm[:, None]
        assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
        feat, _ = np.histogram(assignments, bins=vocab.shape[0])
        feat = feat.astype('float32')
        feat /= np.linalg.norm(feat)
        feats.append(feat)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def nearest_neighbor_classify(train_image_feats,
                              train_labels,
                              test_image_feats,
                              metric='euclidean'):

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    D = sklearn_pairwise.pairwise_distances(test_image_feats,
                                            train_image_feats,
                                            metric="l1")
    idx_D = np.argsort(D, axis=1)
    k = 18
    for x in range(D.shape[0]):
        k_nearest = idx_D[x, :k]
        k_feats = []
        while k or k_feats:
            for idx in k_nearest:
                k_feats.append(train_labels[idx])
            test_labels.append(max(k_feats, key=k_feats.count))
            break

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats,
                 train_labels,
                 test_image_feats,
                 cross_val=False,
                 good_one=False):
    # categories
    categories = list(set(train_labels))
    # construct 1 vs all SVMs for each category
    svms = {
        cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
        for cat in categories
    }

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    if good_one == True:
        svm = LinearSVC(loss='squared_hinge', random_state=0, tol=1e-5, C=1)
        svm.fit(train_image_feats, train_labels)
        test_labels = svm.predict(test_image_feats)
    else:
        n_samples = len(train_labels)
        n_classes = len(categories)
        scores = np.zeros((n_samples, n_classes), dtype=DTYPE)
        for cat, i in zip(categories, range(n_classes)):
            count = 0
            label_ovr = np.zeros((n_samples,))
            for label in train_labels:
                if label == cat:
                    label_ovr[count] = 1
                count += 1
            svms[cat].fit(train_image_feats, label_ovr)
            if cross_val == True:
                cross_validation_svm(svms, cat, train_image_feats, label_ovr, cv=5)
            scores[:, i] = svms[cat].decision_function(test_image_feats)

        for i in range(n_samples):
            test_labels.append(categories[np.argmax(scores[i, :])])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels

def cross_validation_svm(svms, cat, train_image_feats, label_ovr, cv=5):
    temp = cross_val_score(svms[cat], train_image_feats, label_ovr, cv=cv)
    mean = temp.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (temp.mean(), temp.std() * 2))
    C = svms[cat].get_params()['C']
    while mean <= 0.90 and C > 1:
        C -= 0.1
        svms[cat].set_params(C=C, loss='squared_hinge')
        svms[cat].fit(train_image_feats, label_ovr)
        temp = cross_val_score(svms[cat], train_image_feats, label_ovr, cv=cv)
        mean = temp.mean()
        print("Accuracy: %0.2f (+/- %0.2f)" % (temp.mean(), temp.std() * 2))