import cv2
import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from utils import *
import student_code as sc

def plot_results(test_labels, categories, abbr_categories, predicted_categories, fname):
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    # confusion matrix
    y_true = [cat2idx[cat] for cat in test_labels]
    y_pred = [cat2idx[cat] for cat in predicted_categories]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    acc = np.mean(np.diag(cm))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('jet'))
    plt.title('Confusion matrix. Mean of diagonal = {:4.2f}%'.format(acc*100))
    tick_marks = np.arange(len(categories))
    plt.tight_layout()
    plt.xticks(tick_marks, abbr_categories, rotation=45)
    plt.yticks(tick_marks, categories)
    plt.savefig(fname)

categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest']
              
abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst',
                   'Mnt', 'For']

num_train_per_cat = 100
data_path = osp.join('..', 'data')

train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,
                                                                                 categories,
                                                                                 num_train_per_cat)

vocab_set = [10, 20, 50, 100, 200, 400, 1000]
for size in vocab_set:
    fname = 'vocab_'+str(size)
    vocab_size = size
    vocab_filename = fname+'.pkl'
    vocab = sc.build_vocabulary(train_image_paths, vocab_size)
    with open(vocab_filename, 'wb') as f:
        pickle.dump(vocab, f)
        print('{:s} saved'.format(vocab_filename))

    train_image_feats = sc.get_bags_of_sifts(train_image_paths, vocab_filename)
    test_image_feats = sc.get_bags_of_sifts(test_image_paths, vocab_filename)
    predicted_categories = sc.svm_classify(train_image_feats, train_labels, test_image_feats)
    plot_results(test_labels, categories, abbr_categories, predicted_categories, fname+'.png')
    