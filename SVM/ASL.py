import cv2
import numpy as np
import util as ut
import svm_train as st
import re
model=cv2.SVM()
model.load("MODEL.xml")					#change back to 17  -   BHAVYA
#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969
#cam=int(raw_input("Enter Camera number: "))			#BHAVYA
cap=cv2.VideoCapture(0)					 #BHAVYA
font = cv2.FONT_HERSHEY_SIMPLEX

def nothing(x) :
	pass

text= " "

temp=0
previouslabel=None
previousText=" "
label = None
while(cap.isOpened()):
	_,img=cap.read()
	cv2.rectangle(img,(0,0),(400,400),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
	img=np.array(img)
	print img.shape
	img1=img[0:400,0:400,:]
	#print img1
	#img1=img
	img_ycrcb =cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	print img_ycrcb
	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	print blur
	skin_ycrcb_min = np.array([0, 48, 80])
	print skin_ycrcb_min
	skin_ycrcb_max = np.array([20, 255, 255])
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
	contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=ut.getMaxContour(contours,4000)						  # using contours to capture the skin filtered image of the hand
	print cnt
	if type(cnt)!=type(None):
		gesture,label=ut.getGestureImg(cnt,img1,mask,model)
		#label=unichr(64+int(model.predict(st.hog_single(img)) ))  # passing the trained model for prediction and fetching the result
		if(label!=None):
			if(temp==0):
				previouslabel=label
			if previouslabel==label:
				previouslabel=label
				temp+=1
			else :
				temp=0
		if(temp==40):
				if(label=='P'):

					label=" "
				text= text + label
				if(label=='Q'):
					words = re.split(" +",text)
					words.pop()
					text = " ".join(words)
					#text=previousText
				print text

		cv2.imshow('PredictedGesture',gesture)				  # showing the best match or prediction
		cv2.putText(img,label,(50,150), font,8,(0,125,155),2)  # displaying the predicted letter on the main screen
		cv2.putText(img,text,(50,450), font,3,(0,0,255),2)
	cv2.imshow('Frame',img)
	cv2.imshow('Mask',mask)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break



cap.release()
cv2.destroyAllWindows()
