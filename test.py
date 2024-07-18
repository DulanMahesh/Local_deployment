import cv2
img = cv2.imread('Quiz5_1.jpg', cv2.IMREAD_COLOR)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret1, thresh1 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
contours1, hierarchy1 = cv2.findContours( thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

ret2, thresh2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)
contours2, hierarchy2 = cv2.findContours( thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print('Number of contours case 1 : ', len(contours1))
print('Number of contours case 2 : ', len(contours2))
