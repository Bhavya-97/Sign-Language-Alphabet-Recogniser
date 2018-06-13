import cv2
import numpy as np
import util as ut
import svm_train as st
import re
model=st.trainSVM(26)
model.save("MODEL.xml")
