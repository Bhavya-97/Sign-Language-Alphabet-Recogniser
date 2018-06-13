import cv2
import os
import numpy as np
import util as ut
import svm_train as st
import re
model=cv2.SVM()
model.load("MODEL.xml")	
for(direcpath,direcnames,files) in os.walk("./test"):
	for file in files:
		print file
		img=cv2.imread("./test/"+file,0)
		#print img
		
		pred = model.predict(st.hog_single(img))
		#print pred
		print "predicted label: " +unichr(64+int(pred));
		'''#cv2.rectangle(img,(0,0),(400,400),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
		
		#print img
		#img1=img[0:400,0:400,:]
		#print img1
		img1=img
		img_ycrcb =cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
		#print img_ycrcb
		blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
		#print blur
		skin_ycrcb_min = np.array([0, 48, 80],dtype=np.uint8)
		#print skin_ycrcb_min
		skin_ycrcb_max = np.array([20, 255, 255],dtype=np.uint8)
		mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
		contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2)
		cnt=ut.getMaxContour(contours,4000)						  # using contours to capture the skin filtered image of the hand
		#print cnt
		if type(cnt)!=type(None):
			gesture,label=ut.getGestureImg(cnt,img1,mask,model)   # passing the trained model for prediction and fetching the result
			print "Detected Label: "+label+"\n"'''
