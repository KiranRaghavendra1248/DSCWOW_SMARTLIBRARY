#!/usr/bin/env python
from flask import *
import io
import cv2
import numpy as np
from facenet_pytorch import MTCNN
# Imports
import os
import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time
from datetime import date,datetime
import pickle
from PIL import Image
import easyocr
from utils import detect_imgs,show_images,save_data
app = Flask(__name__)
# vc = cv2.VideoCapture(1)
temp__ = None
dum__ = None

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')

@app.route('/home' , methods=["GET", "POST"])
def home():
    global temp__
    if request.method == "POST" :
        usndum = request.form['USN']
        temp__=usndum
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming home page."""
    return render_template('video.html')

@app.route('/sucess')
def sucess():
    """Video streaming home page."""
    return render_template('sucess.html')

@app.route('/unsucess')
def unsucess():
    """Video streaming home page."""
    return render_template('unsucess.html')


@app.route('/scan')
def scan():
    """Video streaming home page."""
    global dum__
    dum__ = "y"
    return render_template("scan.html")

#  def gen():
#     """Video streaming generator function."""
#     while True:
#         read_return_code, frame = vc.read()
#         # print(frame.shape)
#         encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
#         io_buf = io.BytesIO(image_buffer)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

def ben():
    """Video streaming generator function."""
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    # Parameters
    usn_number = temp__
    usn_number = usn_number.strip()
    usn_number = usn_number.upper()
    print(usn_number)
    frame_rate = 16
    prev = 0
    image_size = 600
    threshold = 0.80
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu")
    bbx_color = (0, 255, 0)
    wait_time=10 # For face scan
    time_to_adjust=10 # Before book scan begins

    current_person = None
    # Init MTCNN object
    reader = easyocr.Reader(['en'])  # need to run only once to load model into memory
    mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device, post_process=True)
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    # Real time data from webcam
    frames = []
    boxes = []
    face_results=[]
    # Load stored face data related to respective card number
    faces = []
    usn_nums = []
    face_file = None
    try:
        for usn_ in os.listdir('flask-opencv-streaming-master\Dataset'):
            face_file = open('flask-opencv-streaming-master\Dataset' + '/' + usn_, 'rb')
            if face_file is not None:
                face = pickle.load(face_file)
                faces.append(face)
                usn_nums.append(usn_)
    except FileNotFoundError:
        print('Face data doesnt exist for this USN.')
        exit()
    # Infinite Face Detection Loop
    v_cap = cv2.VideoCapture(0)
    v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
    v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
    flag = False
    start=time.time()
    while (True):
        time_elapsed = time.time() - prev
        break_time=time.time() - start
        if break_time>wait_time:
            break
        ret, frame = v_cap.read()
        if time_elapsed > 1. / frame_rate:  # Collect frames every 1/frame_rate of a second
            prev = time.time()
            frame_ = Image.fromarray(frame)
            frames.append(frame_)
            batch_boxes, prob, landmark = mtcnn.detect(frames, landmarks=True)
            frames_duplicate = frames.copy()
            boxes.append(batch_boxes)
            boxes_duplicate = boxes.copy()
            # show imgs with bbxs
            img,result=show_images(frames_duplicate, boxes_duplicate, bbx_color,transform,threshold,model,faces,usn_nums,usn_number)
            face_results.append(result)
            cv2.imshow("Detection",img)
            frames = []
            boxes = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    v_cap.release()
    cv2.destroyAllWindows()
    accuracy=(sum(face_results)/len(face_results))*100
    print('Percentage match '+'{:.2f}'.format(accuracy))
    if accuracy>60:
        print('Authorization Successful')
        print('Happy Learning')
    else:
        print('Authorization Unsuccessful')
        return render_template('index.html')
        exit()
    temp='y'
    if temp!='y':
        print('No books borrowed')

    
    scan_books(reader,temp,image_size,time_to_adjust,usn_number)

def scan_books(reader,temp,image_size,time_to_adjust,usn_number):
    books=[]
    date_=date.today()
    now = datetime.now()
    time_ = now.strftime("%H:%M:%S")
    while temp=='y':
        print('Image will be captured in 5 sec')
        print('Avoid sudden shaking for better results')
        time.sleep(5)
        v_cap = cv2.VideoCapture(0)
        v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
        v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
        start=time.time()
        while(True):
            curr=time.time()
            if curr-start>=time_to_adjust:
                break
            ret,frame = v_cap.read()
            cv2.imshow('Have a nice day',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.imwrite('Book_img.jpg', frame)
        v_cap.release()
        cv2.destroyAllWindows()
        # Optical character recognition
        book_name=''
        result = reader.readtext('Book_img.jpg')
        for i in result:
            a,b,c=i
            book_name+=' '+b
        if len(book_name) == 0:
            print('No books detected')
        else:
            books.append(book_name)
        #temp=input('Do you wish to scan more books? y/n ')
        temp = 'n'

    if len(books)==0:
        print('No books borrowed')
        return
    print(usn_number+ ' borrowed the following books on '+str(date_)+' at time '+str(time_))
    file1 = open("flask-opencv-streaming-master\myfile.txt", "a")     # append mode
    for i in books:
        file1.write(usn_number+'\t'+i+'\t'+str(date_)+'\t'+str(time_)+'\n')
    file1.close()
    for i in books:
        print(i)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/output_feed')
def output_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        ben(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, threaded=True)