# DSCWOW_SMARTLIBRARY
Library system automation using Real-Time Face Recognition and Optical Character Recogntion

# OBJECTIVE
Automation of  libraries in schools and colleges , for effective storage of records of students who frequent the libraries to borrow books.
Our main aim is to remove the current drawbacks i.e the requirement of a physical presence in the library who takes down records of people borrowing books.
We completely automate the process of borrowing a book so the person who borrows the book has zero or minimal contact with other people.

# IMPLEMENTATION
We use the famous MTCNN algorithm to detect extract and align faces .
The software requires a real time video of the person which spans for 20 seconds as the ground truth value, which is used to train our model.
The student then enters his USN/roll number or anything that uniquely identifies them in their place of study.
The network verifies the person using One Shot Learning using a Siamese Network  when he enters his USN.
We then use OCR (Optical Character Recognition) to automate the process of identifying the name of the book borrowed, 
and in this manner we can store the record of the person borrowing the book.
The record includes the USN of the person borrowing the book , the name of the book and the date and time of when the book was borrowed. 

# HOW TO RUN
1. Git clone/download the repository into the local system.
2. Install torch,torchvision first and the necessary dependencies in requirements.txt next 
using the following commands in a new virtual environment(with python3, 64 bit installed)
```
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.1 -f https://download.pytorch.org/whl/torch_stable.html
```
```
pip install -r requirements.txt
```
3. Run the run_final.py file and make sure to add face data as ground truth value before proceeding to detection.
4. Note: Look at the images in the Result folder for images on how to hold the book for recognition and for sample expected output.
