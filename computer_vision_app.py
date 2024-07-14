import streamlit as st
import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

st.title("COMUTER VISION TASKS")
st.write("### This app  is created to do computer vision tasks which are:")

st.write("#### 1.template matching")
st.write("#### 2.face detection")
st.write("#### 3.object detection")
options=("template matching","face detection", "object detection")
selected_option=st.sidebar.selectbox("select task you want to perform",options)
                    # (1)
if selected_option == "template matching":
    st.write("### template matching")
    
 # Upload image
    uploaded_img = st.file_uploader("Upload full image", type=["jpg", "png", "jpeg"])
    uploaded_template = st.file_uploader("Upload template image", type=["jpg", "png", "jpeg"])
    
    if uploaded_img is not None and uploaded_template is not None:
        # Convert the file to an OpenCV image
        file_bytes_img = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes_img, cv2.IMREAD_COLOR)
        
        file_bytes_template = np.asarray(bytearray(uploaded_template.read()), dtype=np.uint8)
        template = cv2.imdecode(file_bytes_template, cv2.IMREAD_GRAYSCALE)
        
        if img is None or template is None:
            st.error("Error reading one of the images.")
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate width and height of the template
            w, h = template.shape[::-1]

            # Perform template matching
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(result >= threshold)

# draw rectangles
    st.write(f"number of matches found {len(loc[0])}")

    for pt in zip(*loc[::-1]):
     cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,0,255),2)

# show result
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the result using Streamlit
    st.image(img_rgb,caption="detected Matches",use_column_width=True)
                             
                             # (2)
if selected_option == "face detection": 
   st.write("### Face Detection")
    
#  define path to img and file
   uploaded_img = st.file_uploader("Upload full image", type=["jpg", "png", "jpeg"])
   casc_path="haarcascade_frontalface_default.xml"

 # load cascade
   facecascade=cv2.CascadeClassifier(casc_path)

 #read img
   if uploaded_img  is not None:
        # Convert the file to an OpenCV image
        file_bytes_img = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes_img, cv2.IMREAD_COLOR)  
   if img is None:
            st.error("Error reading one of the images.")
   else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
   faces=facecascade.detectMultiScale(
     img_gray,
     scaleFactor=1.05,
     minNeighbors=10,
     minSize=(30,30)
)
   st.write("Found {0} faces!".format(len(faces)))

# draw rectangles
   for x,y,w,h in faces:
     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

# show result
   st.image(img,caption="faces found",use_column_width=True)

                            #(3)
if selected_option == "object detection":
    st.write("### object detection")
    MIN_MATCH_COUNT = 10   

    img1 = st.file_uploader("Upload part to bematched", type=["jpg", "png", "jpeg"])
    img2 = st.file_uploader("Upload full", type=["jpg", "png", "jpeg"])
    # Convert the file to an OpenCV image

    file_bytes_img = np.asarray(bytearray(img1.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes_img, cv2.IMREAD_GRAYSCALE)
        
    file_bytes_template = np.asarray(bytearray(img2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(file_bytes_template, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Find matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,  crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # store all the good matches as per ratio test.
    good = []
    for m in matches:
      if m.distance < 100:
         good.append(m)

   # Draw box and matching keypoints
    if len(good)>MIN_MATCH_COUNT:
       src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
       dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

       M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
       matchesMask = mask.ravel().tolist()

       h,w = img1.shape
       pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
       dst = cv2.perspectiveTransform(pts,M)
       img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
       st.write("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
       matchesMask = None

    draw_params = dict(matchColor = (0,0,255), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
      # Convert the image from BGR to RGB
    img3_rgb=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    st.image(img3_rgb,caption="detected  matches",use_column_width=True)

 