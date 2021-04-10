import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from PIL import Image
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time
import tqdm
from sklearn.model_selection import cross_val_score

def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
        load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
              e.g. if the images are resized to 16x16, d would be 256
    """
    # dummy feats variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for img_path in image_paths:
        img = load_image_gray(img_path)
        feat = cv2.resize(img,(24,24),interpolation=cv2.INTER_AREA).flatten()
        feat_zero_mean = feat-np.mean(feat)
        feat_unit_length = feat_zero_mean/np.linalg.norm(feat_zero_mean)
        feats.append(feat_unit_length)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def build_vocabulary(image_paths, vocab_size):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
            -  frames is a N x 2 matrix of locations, which can be thrown away
            here (but possibly used for extra credit in get_bags_of_sifts if
            you're making a "spatial pyramid").
            -  descriptors is a N x 128 matrix of SIFT features
          Note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster to
          compute. Also, be sure not to use the default value of step size. It
          will be very slow and you'll see relatively little performance gain
          from extremely dense sampling. You are welcome to use your own SIFT
          feature code! It will probably be slower, though.
    -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
            http://www.vlfeat.org/matlab/vl_kmeans.html
              -  X is a N x d numpy array of sampled SIFT features, where N is
                 the number of features sampled. N should be pretty large!
              -  K is the number of clusters desired (vocab_size)
              -  cluster_centers is a K x d matrix of cluster centers. This is
                 your vocabulary.

    Args:
    -   image_paths: list of image paths.
    -   vocab_size: size of vocabulary

    Returns:
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    # length of the SIFT descriptors that you are going to compute.
    dim = 128
    vocab = np.zeros((vocab_size, dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    feats = []
    for i in tqdm.trange(len(image_paths),desc='getting SIFT features'):
        img = load_image_gray(image_paths[i])
        _, descriptors = vlfeat.sift.dsift(img,step=4,fast=True,float_descriptors=True)
        d_norm = np.linalg.norm(descriptors,axis=1)
        idx_nonzero = np.nonzero(d_norm)
        d_norm = d_norm[idx_nonzero]
        descriptors = descriptors[idx_nonzero]
        d_norm = np.linalg.norm(descriptors,axis=1)
        descriptors /= d_norm[:,None]
        feats.append(descriptors)
    
    feats = np.vstack([feat for feat in feats])
    vocab = vlfeat.kmeans.kmeans(feats,vocab_size,
                                 initialization='PLUSPLUS', # RANDSEL, PLUSPLUS
                                 distance='l2', # l1, l2
                                 algorithm='LLOYD') # LLOYD, ELKAN

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab


def get_bags_of_sifts(image_paths, vocab_filename):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
          frames is a M x 2 matrix of locations, which can be thrown away here
            (but possibly used for extra credit in get_bags_of_sifts if you're
            making a "spatial pyramid").
          descriptors is a M x 128 matrix of SIFT features
            note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster
            to compute. Also, be sure not to use the default value of step size.
            It will be very slow and you'll see relatively little performance
            gain from extremely dense sampling. You are welcome to use your own
            SIFT feature code! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
            finds the cluster assigments for features in data
              -  data is a M x d matrix of image features
              -  vocab is the vocab_size x d matrix of cluster centers
              (vocabulary)
              -  assignments is a Mx1 array of assignments of feature vectors to
              nearest cluster centers, each element is an integer in
              [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for i in tqdm.trange(len(image_paths),desc='getting SIFT features'):
        img = load_image_gray(image_paths[i])
#         img = cv2.GaussianBlur(img,(5,5),1,borderType=cv2.BORDER_REPLICATE)
        _, descriptors = vlfeat.sift.dsift(img,step=4,fast=True,float_descriptors=True)
        d_norm = np.linalg.norm(descriptors,axis=1)
        idx_nonzero = np.nonzero(d_norm)
        d_norm = d_norm[idx_nonzero]
        descriptors = descriptors[idx_nonzero]
        descriptors /= d_norm[:,None]
        assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
        feat,_ = np.histogram(assignments,bins=vocab.shape[0])
        feat = feat.astype('float32')
        feat /= np.linalg.norm(feat)
        feats.append(feat)
    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
                              metric='l1'):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   metric: (optional) metric to be used for nearest neighbor.
            Can be used to select different distance functions. The default
            metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
            well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    D = sklearn_pairwise.pairwise_distances(test_image_feats,train_image_feats,metric=metric)
    idx_D = np.argsort(D,axis=1)
    k = 20
    for x in range(D.shape[0]):
        k_nearest = idx_D[x,:k]
        k_feats = []
        for idx in k_nearest:
            k_feats.append(train_labels[idx])
        
        test_labels.append(max(k_feats,key=k_feats.count))
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats, cross_val=False, good_one=False):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.

    Useful functions:
    -   sklearn LinearSVC
          http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    # categories
    categories = list(set(train_labels))
    # construct 1 vs all SVMs for each category
    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
            for cat in categories}

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    if good_one==True:
        svm = LinearSVC(loss='squared_hinge',random_state=0,tol=1e-5,C=1)
        svm.fit(train_image_feats,train_labels)
        test_labels = svm.predict(test_image_feats)
    else:
        n_samples = len(train_labels)
        n_classes = len(categories)
        scores = np.zeros((n_samples,n_classes))
        for cat,i in zip(categories,range(n_classes)):
            count = 0
            label_ovr = np.zeros((n_samples,))
            for label in train_labels:
                if label==cat:
                    label_ovr[count] = 1

                count += 1

            svms[cat].fit(train_image_feats,label_ovr)
            if cross_val==True:
                temp = cross_val_score(svms[cat],train_image_feats,label_ovr,cv=5)
                mean = temp.mean()
                std = temp.std()
                print('Cat ',i)
                print("Accuracy: %0.2f (+/- %0.2f)" % (mean, std * 2))
                C = svms[cat].get_params()['C']
                while mean<=0.90 and C>1:
                    C -= 0.1
                    svms[cat].set_params(C=C,loss='squared_hinge')
                    svms[cat].fit(train_image_feats,label_ovr)
                    temp = cross_val_score(svms[cat],train_image_feats,label_ovr,cv=5)
                    mean = temp.mean()
                    std = temp.std()
                    print("Accuracy: %0.2f (+/- %0.2f)" % (mean, std * 2))

            scores[:,i] = svms[cat].decision_function(test_image_feats)

        for i in range(n_samples):
            idx = np.argmax(scores[i,:])
            test_labels.append(categories[idx])
        
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels
