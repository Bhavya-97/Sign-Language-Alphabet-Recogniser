import cv2
import os
import numpy as np
import util as ut
import svm_train as st
import re
model=cv2.SVM()
model.load("MODEL.xml")

cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
i=1
j=1
name=""

def nothing(x) :
    pass

#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC
    '''Y_min = cv2.getTrackbarPos('Y_min','trackbar')
	Y_max = cv2.getTrackbarPos('Y_max','trackbar')
	Cr_min = cv2.getTrackbarPos('Cr_min','trackbar')
	Cr_max = cv2.getTrackbarPos('Cr_max','trackbar')
	Cb_min = cv2.getTrackbarPos('Cb_min','trackbar')
	Cb_max = cv2.getTrackbarPos('Cb_max','trackbar')
	_,img=cap.read()
	cv2.rectangle(img,(350,128),(600,400),(255,0,0),3)
	img1=img[128:400,350:600]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	skin_ycrcb_min = np.array((Y_min,Cr_min,Cb_min))
	skin_ycrcb_max = np.array((Y_max,Cr_max,Cb_max))
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#ret,mask = cv2.threshold(gray.copy(),20,255,cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=getMaxContour(contours,40)
	if cnt!=None:
		x,y,w,h = cv2.boundingRect(cnt)
		imgT=img1[y:y+h,x:x+w]
		imgT=cv2.bitwise_and(imgT,imgT,mask=mask[y:y+h,x:x+w])
		imgT=cv2.resize(imgT,(300,300))
		cv2.imshow('Trainer',imgT)
	cv2.imshow('Frame',img)
	cv2.imshow('Thresh',mask)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == 13:
		name=str(i)+"_"+str(j)+".jpg"
		cv2.imwrite(name,imgT)
		if(j<20):
			j+=1
		else:
			while(0xFF & cv2.waitKey(0)!=ord('n')):
				j=1
			j=1
			i+=1'''

'''------'''
# Feed the image_data as input to the graph and get first prediction
    

c=0

res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''
alpha=['B','E','H','I','L','O','W','Y','space']
while True:
    ret, img = cap.read()
    
    if ret:
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        c += 1
        image_data = img_cropped#cv2.imencode('.jpg', img_cropped)[1].tostring()
        a = cv2.waitKey(33)
        if i == 4:
            res_tmp= model.predict(st.hog_single(image_data))
            res = alpha[int(res_tmp)-1]
            i = 0
            if mem == res:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive == 10 and res not in ['nothing']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
        i += 1
        cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("img", img)
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('sequence', img_sequence)
    else:
        break
'''------'''
		

cap.release()        
cv2.destroyAllWindows()
