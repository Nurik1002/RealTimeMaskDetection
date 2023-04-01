import tensorflow as tf
import numpy as np
import cvlib
import cv2

def realtime(path = 0):
    """path = 'video_path.mp4'
       path = 0 open camera
    """
    model = tf.keras.models.load_model("model.h5")
    label =  ['WithMask', 'WithoutMask']
    video = cv2.VideoCapture(path)
    while True:
        _, fram = video.read()
        
        faces = cvlib.detect_face(fram)
        
        for x1,y1,x2,y2 in faces[0]:

            face = fram[y1-10:y2+10, x1-10:x2+10] 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)        
            face = cv2.resize(face, (224,224))
            face = np.expand_dims(face, 0)   

            pred, value = np.argmax(model.predict(face)), model.predict(face)
    
            if pred == 0:
                cv2.rectangle(fram, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.putText(fram, f"{label[pred].upper()} : {value[0][1]}" , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            else:                
                cv2.rectangle(fram, (x1,y1), (x2,y2), (0,0,255),2)
                cv2.putText(fram,  f"{label[pred].upper()} : {value[0][0]}", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        
        cv2.imshow("Frame", fram)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    
    if __name__ == "__main__":
      realtime()
