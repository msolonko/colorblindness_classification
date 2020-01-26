import cv2
import numpy as np
class Clusterer:
    def apply(self, image, K):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        vectorized = image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((image.shape))
        return result_image



