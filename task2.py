"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def matcher(key_l_float,key_r_float,desc_l,desc_r):
    
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(desc_r,desc_l,2)
    matches = []
    
    for m in rawMatches:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                
    x= np.float32([key_r_float[i] for (_, i) in matches])
    y= np.float32([key_l_float[i] for (i, _) in matches])
    
    (H, status) = cv2.findHomography(x, y, cv2.RANSAC)
    
    return (H,status,matches)



def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """


    #raise NotImplementedError
    
    l_img=left_img
    r_img=right_img
    
    gr_l= cv2.cvtColor(l_img,cv2.COLOR_BGR2GRAY)
    gr_r= cv2.cvtColor(r_img,cv2.COLOR_BGR2GRAY)


    kaze = cv2.KAZE_create()

    key_l, desc_l = kaze.detectAndCompute(gr_l,None)
    key_r, desc_r= kaze.detectAndCompute(gr_r,None)

    key_l_float = np.float32([i.pt for i in key_l])
    key_r_float = np.float32([i.pt for i in key_r])


    H,s,m=matcher(key_l_float,key_r_float,desc_l,desc_r)

    result = cv2.warpPerspective(r_img, H,(r_img.shape[1] + l_img.shape[1], r_img.shape[0]))

    result[0:l_img.shape[0], 0:l_img.shape[1]] = l_img
    
    return result

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


