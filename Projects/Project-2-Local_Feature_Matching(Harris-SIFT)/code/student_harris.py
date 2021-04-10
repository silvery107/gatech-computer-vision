import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    # 1 compute Ix and Iy
    # 2 compute Ix2, Iy2 and IxIy, then filter them
    # 3 compute Harris corner response matrix by det(H) - 0.05trace(H)^2
    # 4 normalize Harris matrix
    
    # parameter for tuning #
    n_top = 3000 # upper limit for return points
    ANMS = True # adaptive non-maximum supperssion
    AT = True # adaptive thresholding
    up_bound = 5000 # upper bound of # of valid points
    low_bound = 3000 # lower bound of # of valid points
    c_robust = 1 # adaptive non-maximum robust factor
    sigma_d = 1
    sigma_i = 2 # 1-2
    alpha = 0.05
    ratio_p = 1 # ratio of points it process
    ratio_n = 1 # ratio of points it return
    ########################
    
    krnl_size_d = int(np.round(sigma_d*4+1))
    krnl_size_i = int(np.round(sigma_i*4+1))
    img = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)
    m,n = image.shape
    Ix = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=-1,borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=-1,borderType=cv2.BORDER_REPLICATE)
    Ix2 = np.multiply(Ix,Ix)
    Iy2 = np.multiply(Iy,Iy)
    IxIy = np.multiply(Ix,Iy)
    Ix2 = cv2.GaussianBlur(Ix2,(krnl_size_i,krnl_size_i),sigma_i,cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(Iy2,(krnl_size_i,krnl_size_i),sigma_i,cv2.BORDER_REPLICATE)
    IxIy = cv2.GaussianBlur(IxIy,(krnl_size_i,krnl_size_i),sigma_i,cv2.BORDER_REPLICATE)
    det = np.multiply(Ix2,Iy2)-np.multiply(IxIy,IxIy)
    tr2 = (Ix2+Iy2)*(Ix2+Iy2)
    img_Harris = det-alpha*tr2
    img_Harris = cv2.normalize(img_Harris,None,0,255,norm_type=cv2.NORM_MINMAX)
    img_Harris = np.array(img_Harris).astype('uint8')
#     plt.imshow(img_Harris,'gray')
#     plt.show()
    
    #############################################################################

    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # 1 compute or set the threshold for Harris matrix
    # 2 get intresting points coordinates and sort them by response
    # 3 compute the minimum distance from each pixel to its stronger neighbors
    # 4 sort intresting points by minimum distance
    # 5 return top n points
    
    # adaptive threshold
    thr, img_thr = cv2.threshold(img_Harris,-1,255,cv2.THRESH_OTSU)
    x,y = np.nonzero(img_thr)
    while AT==True and (len(x)<low_bound or len(x)>up_bound):
        if thr>254 or thr<1:
            break
        elif len(x)<low_bound:
            thr -= 1
            _,img_thr = cv2.threshold(img_Harris,thr,255,cv2.THRESH_BINARY)
            x,y = np.nonzero(img_thr)
        elif len(x)>up_bound:
            thr += 1
            _,img_thr = cv2.threshold(img_Harris,thr,255,cv2.THRESH_BINARY)
            x,y = np.nonzero(img_thr)
#     else:
#             plt.imshow(img_thr,'gray')
#             plt.show()

    # suppress boundary points
    step = feature_width//2
    for i in range(len(x)):
        if x[i]<=step or x[i]>=m-step or y[i]<=step or y[i]>=n-step:
            x[i]=0
            y[i]=0
    x = x[np.nonzero(x)[0]]
    y = y[np.nonzero(y)[0]]
    
    p_len = x.size # number of valid intresting points  
    p_num = int(ratio_p*p_len)
    n_num = int(ratio_n*p_num) if ratio_n*p_len<n_top else n_top
    points = np.zeros((4,p_len))
    points[0,:] = x
    points[1,:] = y
    for i in range(p_len):
        points[2,i] = img_Harris[x[i],y[i]]

    points = points[:,points[2].argsort()] # sort by cornel response
    points = points[:,p_len-p_num:] # strink to ratio_p
    
    # adaptive non-maximum suppression
    if ANMS==True:
        for p in range(p_num-1):
            r_min = np.Inf
            for q in range(p+1,p_num):
                cur_x1 = points[:2,p].astype('int32')
                cur_x2 = points[:2,q].astype('int32')
                if img_Harris[cur_x1[0],cur_x1[1]]<c_robust*img_Harris[cur_x2[0],cur_x2[1]]:
                    r = np.linalg.norm(cur_x1-cur_x2)
                    if r < r_min:
                        r_min = r
            points[3,p] = r_min
        points = points[:,points[3].argsort()] # sort by min radius
    points = points[:,::-1]
    x = points[1,:n_num].astype('int32') # return top n points
    y = points[0,:n_num].astype('int32') # return top n points
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations


