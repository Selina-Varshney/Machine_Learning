import os
import cv2


sample = cv2.imread("Altered/Altered-Hard/150__M_Left_middle_finger_Zcut.BMP")
sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

cv2.imshow("Sample", sample)
cv2.waitKey(0)
cv2.destroyAllWindows()

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None
for file in [file for file in os.listdir("Real")][:100]:
    finger_print = cv2.imread("Real/" + file)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(finger_print, None)


    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10,},
                                    {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []

    for p,q in matches:
        if p.distance < 0.2 * q.distance:
            match_points.append(p)
 
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    if len(match_points) / keypoints * 100 > best_score:
        best_score = (len(match_points) / keypoints * 100)
        filename = file
        image = finger_print
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

if(filename == None):
    print("Not Identified")
    print("UNAUTHORIZED USER")
    img1 = cv2.imread("No_entry.png")
    img1 = cv2.resize(img1, None, fx=2.5, fy=2.5)
    cv2.imshow("Noentry", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


else:
    print("IDENTIFIED MATCH: " + filename)
    print("AUTHORIZED USER")
    img2 = cv2.imread("authorized.jpg")
    img2 = cv2.resize(img2, None, fx=2, fy=2)
    cv2.imshow("Authorized", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print("SCORE: " + str(best_score))
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result =cv2.resize(result, None, fx=2, fy=2)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



