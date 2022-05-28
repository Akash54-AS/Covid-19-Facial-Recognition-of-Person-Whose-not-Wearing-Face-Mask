
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
from os import listdir
import face_recognition




#put your employee pictures in this path as name_of_employee.jpg
person_pictures = r'C:\Users\Admin\Desktop\Machine Learning\face_detector\dataset'

persons = []
names=[]
for file in listdir(person_pictures):
	person, extension = file.split(".")
	img = face_recognition.load_image_file(r'C:\Users\Admin\Desktop\Machine Learning\face_detector\dataset\%s.jpeg' % (employee))
	embeddings =face_recognition.face_encodings(img)[0] 
	names.append(person)
	persons.append(embeddings)
	
print("person representations retrieved successfully")

#------------------------
# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.70:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) >0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		print(preds)
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["C:\\Users\\Admin\\Desktop\\Machine Learning", "deploy.prototxt"])
weightsPath = os.path.sep.join([r"C:\Users\Admin\Desktop\Machine Learning",
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("C:\\Users\\Admin\\Desktop\\Machine Learning\\face_detector\\training_model")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
while(True) :
    ret,f=vs.read()
    img=f
    frame =imutils.resize(f,width=400)
    
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
    
    (locs,preds) =detect_and_predict_mask(frame, faceNet, maskNet)
    #print(preds)   
    for(box,pred,encode) in zip(locs,preds,face_encodings):
        (startX,startY,endX,endY)=box
        (mask,withoutMask) =pred
        #print(pred)        
        label="Mask" if mask>withoutMask else "No Mask"
        color =(0,255,0) if label == "Mask" else (0,0,255)
        
        label ="{}: {:.2f}%".format(label, max(mask,withoutMask)*100)
        
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        if mask<withoutMask and locs!={}:
           
            name ="unknown"
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(persons, encode)
    
            print(matches)
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]

    
            print(name)    
            # Display the results
            #print(face_locations)
            (x1,y1,x2,y2)=box
            cv2.putText(frame, name, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,0.6, (256,0,0), 2)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)& 0xFF

    if key==ord("q"):
        break
vs.release()
cv2.destroyAllWindows()   