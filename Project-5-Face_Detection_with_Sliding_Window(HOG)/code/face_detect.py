import cv2
import numpy as np
import matplotlib.pyplot as plt
import cyvlfeat as vlfeat
import os.path as osp
from utils import *
from glob import glob
from sklearn.svm import LinearSVC,SVC
from utils import *

def get_positive_features(train_path_pos, feature_params):
    """
    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    block_size = 2*cell_size
    block_stride = cell_size
    hog = cv2.HOGDescriptor((win_size,win_size),
                            (block_size,block_size),
                            (block_stride,block_stride),
                            (cell_size,cell_size),9)
    
    feats = []
    for path in positive_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # feat = vlfeat.hog.hog(img, cell_size)
        feat = hog.compute(img).reshape(-1)
        feats.append(feat)
        img = cv2.flip(img,1)
        # feat = vlfeat.hog.hog(img, cell_size)
        feat = hog.compute(img).reshape(-1)
        feats.append(feat)
    
    feats = np.array(feats)#.reshape(len(feats),-1)

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    block_size = 2*cell_size
    block_stride = cell_size
    hog = cv2.HOGDescriptor((win_size,win_size),
                            (block_size,block_size),
                            (block_stride,block_stride),
                            (cell_size,cell_size),9)
    feats = []
    count = 0
    step = 1
    for path in negative_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # feat = vlfeat.hog.hog(img,cell_size)
        feat = hog.compute(img).reshape(-1)
        # feat = np.array(feat)
        f_shape = feat.shape
        for x in range(0,f_shape[0]-900,step):
            feats.append(feat[x:x+900])
        # for x in range(0,f_shape[0]-cell_size,step):
        #     for y in range(0,f_shape[1]-cell_size,step):
        #         feats.append(feat[x:x+cell_size,y:y+cell_size].reshape(-1))
                
        count += len(feats)
        if count>num_samples:
            break
            
    feats = np.array(feats)

    return feats

def train_classifier(features_pos, features_neg, C):
    """
    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """

    n_pos = features_pos.shape[0]
    n_neg = features_neg.shape[0]
    svm = LinearSVC(C=C)
    feats = np.vstack((features_pos,features_neg))
    labels = np.hstack((np.ones(n_pos,),-np.ones(n_neg,)))
    svm.fit(feats,labels)

    return svm

def run_detector(test_scn_path, svm, feature_params, verbose=False):
   
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    # number of top detections to feed to NMS
    topk = 15

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.7)
    template_size = int(win_size / cell_size)

    # im = cv2.cvtColor(test_scn_path,cv2.COLOR_RGB2GRAY)
    im = test_scn_path
    im_shape = im.shape

    # Parameters for tuning
    step = 1 # 1 cell per step, namely 6 ptx
    score_thr = 0
    scale_topk = 40
    scale_flag = True
    BF_flag = False

    cur_bboxes = np.empty((0, 4))
    cur_confidences = np.empty(0)
    scale_times = 0
    scale_shape = im_shape

    cur_bboxes,cur_confidences = hog.detectMultiScale(im)
    # while min(scale_shape)>win_size:
    #     feat = vlfeat.hog.hog(im,cell_size)
    #     feat = np.array(feat)
    #     f_shape = feat.shape
    #     temp_bb = np.empty((0, 4))
    #     temp_con = np.empty(0)
    #     for x in range(0,f_shape[0]-template_size,step):
    #         for y in range(0,f_shape[1]-template_size,step):
    #             feat_ROI = feat[x:x+template_size,y:y+template_size]
    #             score = svm.decision_function(feat_ROI.reshape(1,-1))
    #             if score>score_thr:
    #                 cur_x = round(template_size*x/scale_factor**scale_times)
    #                 cur_y = round(template_size*y/scale_factor**scale_times)
    #                 cur_size = round(win_size/scale_factor**scale_times)
    #                 temp_bb = np.vstack((temp_bb,[cur_y,cur_x,cur_y+cur_size,cur_x+cur_size]))
    #                 temp_con = np.hstack((temp_con,score[0]))
        
    #     if BF_flag == False:
    #         idsort = np.argsort(-temp_con)[:scale_topk]
    #         temp_bb = temp_bb[idsort]
    #         temp_con = temp_con[idsort]
        
    #     if temp_bb.shape[0] != 0:
    #         is_valid_bbox = non_max_suppression_bbox(temp_bb, temp_con, im_shape, verbose=verbose)
    #         temp_bb = temp_bb[is_valid_bbox]
    #         temp_con = temp_con[is_valid_bbox]
    #         cur_bboxes = np.vstack((cur_bboxes,temp_bb))
    #         cur_confidences = np.hstack((cur_confidences,temp_con))
            
    #     im = cv2.resize(im,(round(scale_shape[1]*scale_factor),round(scale_shape[0]*scale_factor)),cv2.INTER_AREA)
    #     scale_shape = im.shape
    #     scale_times += 1
    #     if scale_flag == False:
    #         scale_shape = (win_size,win_size)
        
    if len(cur_bboxes) == 0:
        return None,None
            
    # idsort = np.argsort(-cur_confidences)[:topk]
    # cur_bboxes = cur_bboxes[idsort]
    # cur_confidences = cur_confidences[idsort]
    # is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
    #     im_shape, verbose=verbose)

    # print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
    # cur_bboxes = cur_bboxes[is_valid_bbox]
    # cur_confidences = cur_confidences[is_valid_bbox]

    bboxes = np.vstack((bboxes, cur_bboxes)).astype(np.int32)
    confidences = np.hstack((confidences, cur_confidences))

    return bboxes, confidences

def hog_detect_mul(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    win_size = 36
    cell_size = 6
    block_size = 2*cell_size
    block_stride = cell_size
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img)
    return rects


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")