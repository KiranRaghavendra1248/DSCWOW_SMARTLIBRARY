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

def take_input():
    image_size = 600
    frame_rate = 64
    vid_len = 20  # Length of video in seconds
    usn_number = input('Enter the USN Number: ').strip()
    usn_number=usn_number.upper()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu")


    v_cap = cv2.VideoCapture(0)
    v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
    v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
    count = 1
    prev = 0
    try:
        os.mkdir('Dataset')
    except FileExistsError:
        pass

    mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device, post_process=True)
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    start = time.time()
    frames = []
    boxes = []
    print('Try to sit in a well lit position')
    time.sleep(3)
    print(
        'Try to keep your face at the centre of the screen and turn ur face slowly in order to capture diff angles of your face')
    time.sleep(3)
    print('A window will pop up any time now')
    save_tensor = None

    # 20 sec loop to input truth face images
    while True:
        time_elapsed = time.time() - prev
        curr = time.time()
        if curr - start >= vid_len:
            break
        ret, frame = v_cap.read()
        cv2.imshow('Recording ', frame)
        if time_elapsed > 1. / frame_rate:  # Collect frames every 1/frame_rate of a second
            prev = time.time()
            frame_ = Image.fromarray(frame)
            frames.append(frame_)
            batch_boxes, prob, landmark = mtcnn.detect(frames, landmarks=True)
            frames_duplicate = frames.copy()
            boxes.append(batch_boxes)
            boxes_duplicate = boxes.copy()
            # show imgs with bbxs
            if save_tensor == None:
                save_tensor = save_data(frames_duplicate, boxes_duplicate,model)
            else:
                temp = save_data(frames_duplicate, boxes_duplicate,model)
                if temp is not None:
                    save_tensor = torch.cat([temp, save_tensor], dim=0)
                    print(save_tensor.shape)
            count += 1
            frames = []
            boxes = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Open file for pickling
    face_file = open('Dataset' + '/' + usn_number, 'ab')
    pickle.dump(save_tensor, face_file)
    face_file.close()
    v_cap.release()
    cv2.destroyAllWindows()


def detection():
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    # Parameters
    usn_number = input('Enter the USN : ').strip()
    usn_number = usn_number.upper()
    frame_rate = 16
    prev = 0
    image_size = 600
    threshold = 0.80
    device = device = torch.device(
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
        for usn_ in os.listdir('Dataset'):
            face_file = open('Dataset' + '/' + usn_, 'rb')
            if face_file is not None:
                face = pickle.load(face_file)
                faces.append(face)
                usn_nums.append(usn_)
    except FileNotFoundError:
        print('Dataset folder is corrupted')
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
            cv2.imshow('Detection',img)
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
        exit()

    temp=input('Start scan for books? y/n ')

    if temp!='y':
        print('No books borrowed')
        return
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
        temp=input('Do you wish to scan more books? y/n ')

    if len(books)==0:
        print('No books borrowed')
        return
    print(usn_number+ ' borrowed the following books on '+str(date_)+' at time '+str(time_))
    file1 = open("myfile.txt", "a")     # append mode
    for i in books:
        file1.write(usn_number+'\t'+i+'\t'+str(date_)+'\t'+str(time_)+'\n')
    file1.close()
    for i in books:
        print(i)

if __name__=='__main__':
    print('1 to add face_data')
    print('2 for detection & scanning')
    temp=input()
    if temp=='1':
        take_input()
    elif temp=='2':
        detection()
    else:
        exit()


