from cvzone.FaceDetectionModule import FaceDetector
import cv2 
import cvzone
from time import time

###########################################
classID = 1   #0 is for fake and 1 is for real       # 2 - mask    # 3 - specs     (data should be in new folder)   
outputFolderPath = "/Users/apuravjain/Desktop/Hackathon/Real_ImagesCapstone"              # Target --> ( video, eyes blink, phone(closure),captcha )
confidence=0.8
save=True                                        # real_imagescapstone is a folder for collecting the images in demonstration of the project.
blurThreshold = 35 #Larger is more focused

offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight= 640,480
floatingPoint = 6
debug=False
##########################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4, camHeight)
detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  #Contains true false values indicating if the faces are blur or not
    listInfo = []  #The normalised values and the class name or the labeled text file

    if bboxs:

        for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
            score=bbox["score"][0]
            #print(x,y,w,h)

            #Check the score
            if float(score)>confidence:
                
                #Adding an offset ot the face detected
                offsetW = (offsetPercentageW/100)*w
                x=int(x-offsetW)
                w=int(w+offsetW*2)

                offsetH = (offsetPercentageH/100)*h
                y=int(y-offsetH*3)
                h=int(h+offsetH*4.5)

                #To avoid values below zero
                x=max(x,0)
                y=max(y,0)
                h=max(h,0)
                w=max(w,0)

                #Find bluriness in the face
                imgFace = img[y:y+h, x:x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue>blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                #Normalizing the values
                imgH, imgW, channel = img.shape
                xcenter, ycenter = x+w/2, y+h/2
                xnorm, ynorm = round(xcenter/imgW,floatingPoint), round(ycenter/imgH, floatingPoint)
                wnorm, hnorm = round(w/imgW,floatingPoint), round(h/imgH, floatingPoint)
                #print(xnorm, ynorm, wnorm, hnorm)

                #To avoid values above one
                xnorm=min(xnorm,1)
                ynorm=min(ynorm,1)
                hnorm=min(hnorm,1)
                wnorm=min(wnorm,1)

                listInfo.append(f"{classID} {xnorm} {ynorm} {wnorm} {hnorm}\n")  #Argument of append is format that YOLO requires

                #Drawing
                cv2.rectangle(imgOut, (x,y,w,h), (255,0,0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blurValue}', {x,y-20}, scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score*100)}% Blur: {blurValue}', {x,y-20}, scale=2, thickness=3)
        #To Save
        if save:
            if all(listBlur) and listBlur!=[]:
                #Save image
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0]+timeNow[1]
                #print(timeNow)
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)

                #Save label text file
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", "a")
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)