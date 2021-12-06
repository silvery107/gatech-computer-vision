# Computer Vision Assignment Report

Title: Scene recognition with bag of words

Student Name: Mandan Chao

Student ID: 11811737

### 1. Experimental Design
- Use cross-validation to measure performance rather than the fixed test / trainsplit provided by the starter code. Randomly pick 100 training and 100 testingimages for each iteration and report average performance and standard-deviations.

```python
def cross_validation_svm(svm, train_image_feats, label_ovr, cv=5):
    temp = cross_val_score(svm, train_image_feats, label_ovr, cv=cv)
    mean = temp.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (temp.mean(), temp.std() * 2))
```

- Add a validation set to your training process to tune learning parameters. This validation set could either be a subset of the training set or some of theotherwise unused test set.

```python
def auto_tune_svm(svm, train_image_feats, label_ovr, cv):
    C = svm.get_params()['C']
    while mean <= 0.90 and C > 1:
        C -= 0.1
        svm.set_params(C=C, loss='squared_hinge')
        svm.fit(train_image_feats, label_ovr)
        temp = cross_val_score(svm, train_image_feats, label_ovr, cv=cv)
        mean = temp.mean()
        print("Accuracy: %0.2f (+/- %0.2f)" % (temp.mean(), temp.std() * 2))
```

- Experiment with many different vocabulary sizes and report performance. E.g.10, 20, 50, 100, 200, 400, 1000, 10000.




### 2. Experimental Results Analysis

- Cross Validation

- Learning Parameters Tuning

- Vocabulary Size Analysis





### 3. Bonus Report (If you have done any bonus problem, state them here)

- Accelerate the process of generating sift features


- Add cross validation and the accuracy of three combinations exceeds the average

