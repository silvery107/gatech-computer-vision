import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    # 1 get the feature patch based on intresting points
    # 2 soomth with a large Gaussion filter
    # 3 calculate the gradient histogram in each quarter of a patch, which totally gives a 4*4*8 dim vector
    # 4 normalize it, clip it with 0.2 and renormalize it
    
    assert feature_width%4 == 0
    x,y = y,x
    img = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)
    
    # parameter for tuning #
    c_power = 1 # 0.8-1
    clip_tail = 0.2
    ########################
    
    # suppress boundary points
    m,n = img.shape
    step = feature_width//2
    for i in range(len(x)):
        if x[i]<=step or x[i]>=m-step or y[i]<=step or y[i]>=n-step:
            x[i]=0
            y[i]=0
    
    x = x[np.nonzero(x)[0]]
    y = y[np.nonzero(y)[0]]
    
    feat_num = len(x)
    step = feature_width//2
    
    # feature by normalized image patches
#     fv = np.zeros((feat_num,feature_width**2))
#     for i in range(feat_num):
#         feature = (img[x[i]-step:x[i]+step,y[i]-step:y[i]+step]).flatten()
#         fv[i] = np.array(cv2.normalize(feature,None,norm_type=cv2.NORM_MINMAX)).flatten()

    # feature by SIFT-like descriptor
    fv = np.zeros((feat_num,feature_width**2//2))
    for i in range(feat_num):
        feature = img[x[i]-step:x[i]+step+1,y[i]-step:y[i]+step+1]
        Ix = cv2.Sobel(feature,cv2.CV_32F,1,0,-1,borderType=cv2.BORDER_DEFAULT)
        Iy = cv2.Sobel(feature,cv2.CV_32F,0,1,-1,borderType=cv2.BORDER_DEFAULT)
        G,theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
        g_krnl = cv2.getGaussianKernel(ksize=feature_width+1,
                                       sigma=(feature_width+1)//2,
                                       ktype=cv2.CV_32F)
        g_krnl = g_krnl*g_krnl.T
        G = np.multiply(G,g_krnl)
        hist = np.zeros((feature_width,8))
        for j in range(0,feature_width,feature_width//4):
            for k in range(0,feature_width,feature_width//4):
                # 5*5 points, 4*4 cells
                temp_hist = np.histogram(theta[j:j+5,k:k+5].flatten(),
                                        bins=8,
                                        range=(0,360),
                                        weights=G[j:j+5,k:k+5].flatten()
                                        )[0]
                hist[j+k//4,:] = temp_hist

        temp = np.hstack([h for h in hist])
        temp /= np.linalg.norm(temp)
        temp = np.clip(temp,0,clip_tail)
        fv[i] = (temp/np.linalg.norm(temp))**c_power
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
# trash

#     o_size = feature_width # 9*1 or 9*2
#     o_step = int(o_size//2)
#         G = np.zeros((feature_width+1,feature_width+1))
#         theta = np.zeros((feature_width+1,feature_width+1))
#         if ORIEN == True:
#             # compute the main orientation of keypoint
#             o_G = img_G[x[i]-o_step:x[i]+o_step+1,y[i]-o_step:y[i]+o_step+1]
#             o_theta = img_theta[x[i]-o_step:x[i]+o_step+1,y[i]-o_step:y[i]+o_step+1]
#             g_krnl = cv2.getGaussianKernel(ksize=o_G.shape[0],sigma=1.5*sigma,ktype=cv2.CV_32F)
#             g_krnl = g_krnl*g_krnl.T
#             G = np.multiply(o_G,g_krnl)
#             hist,bins = np.histogram(o_theta.flatten(),
#                                      bins=36,range=(0,360),
#                                      weights=o_G.flatten())
#             h_smooth = np.array([1,4,6,4,1])/16
#             hist = np.convolve(hist,h_smooth,'same')
#             argm = np.argmax(hist)
#             orientations = int(bins[argm]+5)

#             # rotate feature patch back to the main orientation of keypoint
#             ori = -orientations
#             feature_ex = img[x[i]-r:x[i]+r+1,y[i]-r:y[i]+r+1]
#             M = cv2.getRotationMatrix2D((r,r), ori, 1)
#             feature_rotated = cv2.warpAffine(feature_ex, M, feature_ex.shape)
            
#             # 17*17 points, 16*16 patches
#             feature = feature_rotated[r-step:r+step+1,r-step:r+step+1]
#             Ix = cv2.Sobel(feature,cv2.CV_32F,1,0,-1,borderType=cv2.BORDER_REPLICATE)
#             Iy = cv2.Sobel(feature,cv2.CV_32F,0,1,-1,borderType=cv2.BORDER_REPLICATE)
#             G,theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)

#         else:
#             G = img_G[x[i]-step:x[i]+step+1,y[i]-step:y[i]+step+1]
#             theta = img_theta[x[i]-step:x[i]+step+1,y[i]-step:y[i]+step+1]
          