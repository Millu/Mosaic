import numpy as np
import cv2

CAM_KX = 720.
CAM_KY = CAM_KX
CAM_CX = 320.
CAM_CY = 200.

# DO NOT MODIFY cam_params_to_mat
def cam_params_to_mat(cam_kx, cam_ky, cam_cx, cam_cy):
    """Returns camera matrix K (3x3 numpy.matrix) from the focus and camera center parameters.
    """
    K = np.reshape(np.mat([cam_kx, 0, cam_cx, 0, cam_ky, cam_cy, 0, 0, 1]), (3,3))
    return K

# DO NOT MODIFY descript_keypt_extract
def descript_keypt_extract(img):
    """Takes an image and converts it to a list of descriptors and corresponding keypoints.

    From http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb

    Returns:
        des (np.ndarray): Nx32 array of N feature descriptors of length 32.
        kpts (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        img2 (np.array): Image which draws the locations of found keypoints.
    """

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kpts = orb.detect(img,None)

    # compute the descriptors with ORB
    kpts, des = orb.compute(img, kpts)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kpts, color=(255,0,0), flags=0)

    kpts = [np.mat([kpt.pt[0], kpt.pt[1], 1.0]).T for kpt in kpts]

    return des, kpts, img2

# fill your in code here
def propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b):
    """Given a set of descriptors and keypoints from two images, propose good keypoint pairs.

    Feature descriptors should encode local image geometry in a way that is invariant to
    small changes in perspective. They should be comparible using an L2 metric.

    For the top N matching descrpitors, select and return the top corresponding keypoint pairs.

    Returns:
        pair_pts_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        pair_pts_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here
    pair_pts_a = []
    pair_pts_b = []
    
    for i in range(len(descripts_a)): 
        temp = []
        for j in range(len(descripts_b)):
            ad = descripts_a[i] 
            bd = descripts_b[j]                
            dist = cv2.norm(ad,bd,cv2.NORM_HAMMING)
            temp.append([dist, j])

        temp = sorted(temp)
        pair_pts_a.append(keypts_a[i])
        pair_pts_b.append(keypts_b[temp[0][1]])
    #print "A: \n", pair_pts_a
    #print "B: \n", pair_pts_b

    return pair_pts_a, pair_pts_b

# fill your in code here
def homog_dlt(ptsa, ptsb):
    """From a list of point pairs, find the 3x3 homography between them.

    Find the homography H using the direct linear transform method. For correct
    correspondences, the points should all satisfy the equality
    w*ptb = H pta, where w > 0 is some multiplier. 

    Arguments:
        ptsa (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        ptsb (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT.
    """
    # code here
    a = []
    print ptsa
    for i in range(len(ptsa)):
        u, v = ptsa[i, 0], ptsa[i, 1]
        u_p, v_p = ptsb[i, 0], ptsb[i, 1]
        a.append([u, v, 1, 0, 0, 0, -u_p*u, -u_p*v, -u_p])
        a.append([0, 0, 0, u, v, 1, -v_p*u, -v_p*v, -v_p])
    a = np.asarray(a)
    u, s, vt = np.linalg.svd(a)
    v = vt.T
    h = v[:,-1]
    res = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]]
    H = np.asarray(res)
    return H

# fill your in code here
def homog_ransac(pair_pts_a, pair_pts_b):
    """From possibly good keypoint pairs, determine the best homography using RANSAC.

    For a set of possible pairs, many of which are incorrect, determine the homography
    which best represents the image transformation. For the best found homography H,
    determine which points are close enough to be considered inliers for this model
    and return those.

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT and RANSAC.
        best_inliers_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        best_inliers_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here

    # H = homog_dlt(x,y)

    return best_H, best_inliers_a, best_inliers_b

# DO NOT MODIFY perspect_combine
def perspect_combine(img_a, img_b, H, length, width):
    """Perspective warp and blend two images based on given homography H.

    Create img_ab by first warping all the pixels in img_a according to the relation
    img_b = H img_a.  For pixels in both img_a and img_b, average the intensity.
    Otherwise, pick the image value present, and black otherwise.
    """
    bw = img_b.shape[0]
    bl = img_b.shape[1]

    warp_a = cv2.warpPerspective(img_a, H, (length, width))
    mask_a = np.zeros((bw, bl, 3))
    for i in range(bw):
        for j in range(bl):
            if np.sum(warp_a[i][j]) > 0.:
                mask_a[i,j,:] = 1.
    img_ab = warp_a
    img_ab[:bw,:bl,:] -= warp_a[:bw,:bl,:]/2.*mask_a[:bw,:bl,:]
    img_ab[:bw,:bl,:] += img_b*(1-mask_a[:bw,:bl,:])
    img_ab[:bw,:bl,:] += img_b/2.*mask_a[:bw,:bl,:]
    return img_ab

# DO NOT MODIFY img_combine_homog
def img_combine_homog(img_a, img_b, length_ab, width_ab):
    """Perspective warp and blend two images of nearby perspectives.
    """
    descripts_a, keypts_a, img_keypts_a = descript_keypt_extract(img_a)
    print descripts_a
    descripts_b, keypts_b, img_keypts_b = descript_keypt_extract(img_b)

    if True:
        cv2.imshow('Keypts A', img_keypts_a)
        cv2.imshow('Keypts B', img_keypts_b)

    pair_pts_a, pair_pts_b = propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b)
    assert(len(pair_pts_a) == len(pair_pts_b))
    assert(np.shape(pair_pts_a[0]) == (3,1) and np.shape(pair_pts_b[0]) == (3,1))
    assert(isinstance(pair_pts_a[0], np.matrix) and isinstance(pair_pts_b[0], np.matrix))

    best_H, best_inliers_a, best_inliers_b = homog_ransac(pair_pts_a, pair_pts_b)
    assert(np.shape(best_H) == (3,3) and isinstance(best_H, np.matrix))
    print len(best_inliers_a)

    fixed_H = homog_dlt(best_inliers_a, best_inliers_b)
    assert(np.shape(fixed_H) == (3,3) and isinstance(fixed_H, np.matrix))

    img_ab = perspect_combine(img_a, img_b, fixed_H, length_ab, width_ab)
    return img_ab, fixed_H

# fill your in code here
def rot_from_homog(H, K):
    """Find the rotation matrix from a homography from perspectives with identical camera centers.
    
    The rotation found should be bRa or Ra^b.

    Arguments:
        H (np.matrix of shape 3x3): Homography
        K (np.matrix of shape 3x3): Camera matrix
    Returns:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    """
    # code here
    return R


# fill your in code here
def extract_y_angle(R):
    """Given a rotation matrix around the y-axis, find the angle of rotation.

    The matrix need not be perfectly in SO(3), but provides an estimate nonetheless.

    Arguments:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    Returns:
        y_ang (float): angle in radians
    """
    # code here
    return y_ang



# DO NOT MODIFY single_pair_combine
def single_pair_combine(img_ai, img_bi):
    img_a = cv2.imread('image_%02d.png'%img_ai)
    img_b = cv2.imread('image_%02d.png'%img_bi)

    # decimate by 2
    img_a = img_a[::2,::2,:]
    img_b = img_b[::2,::2,:]

    length_ab, width_ab = 1000, 600
    img_ab, best_H = img_combine_homog(img_a, img_b, length_ab, width_ab)

    K = cam_params_to_mat(CAM_KX, CAM_KY, CAM_CX, CAM_CY)
    R = rot_from_homog(best_H, K)
    assert(np.shape(R) == (3,3) and isinstance(R, np.matrix))
    y_ang = extract_y_angle(R)

    print 'H'
    print best_H
    print 'K'
    print K
    print 'R'
    print R
    print 'Y angle in Radians/Degrees'
    print y_ang, np.rad2deg(yang)

    cv2.imshow('Image A', img_a)
    cv2.imshow('Image B', img_b)
    cv2.imshow('Combined Image', img_ab)
    cv2.waitKey(0)

# DO NOT MODIFY multi_pair_combine
def multi_pair_combine(beg_i, n_imgs):
    img_ab = cv2.imread('image_%02d.png'%beg_i)
    img_ab = img_ab[::2,::2,:]
    for i in range(beg_i+1, beg_i+n_imgs):
        img_b = cv2.imread('image_%02d.png'%i)

        # decimate by 2
        img_b = img_b[::2,::2,:]

        length_ab, width_ab = 800+400*(i-1), 600
        img_ab, best_H = img_combine_homog(img_ab, img_b, length_ab, width_ab)

        cv2.imshow('Combined Image', img_ab)
        cv2.waitKey(100)
    cv2.imwrite('combo_%02d_%02d.png'%(beg_i, beg_i+n_imgs-1), img_ab)

# DO NOT MODIFY main
def main():
    if True:
        single_pair_combine(0, 1)

    if False:
        multi_pair_combine(0, 5)

#if __name__ == "__main__":
#    main()

