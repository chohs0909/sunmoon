import tkinter as tk
from tkinter import *
import cv2
import time
import tensorflow as tf
import numpy as np
import threading
from PIL import Image, ImageTk


window = tk.Tk()
#tot_text를 전역변수로 선언
global count
global tot_text
#tot_text 초기화
tot_text = []
count = 0
window.geometry('1280x720')
window.title("수어번역기")
label = tk.Label(window)
label.place(x=0, y=0)



frame4 = Frame(window,width=1280,height=720)
frame6 = Frame(window,width=1280,height=720)
frame4.grid(row=0,column=0, )
frame4.propagate(0)
frame6.grid(row=0,column=0, )
frame6.propagate(0)
video_label = tk.Label(frame4)
video_label.pack(pady=10)
#ptrdicted_class_label 초기화
predicted_class_label = ""


def openFrame(frame):
    frame.tkraise()
    if frame == frame4:
        signlanguage()
def process_frame():
    global predicted_class_label
    global class_labels
    global count
    global tot_text
    global encText

    # 모델 적용
    model_path = 'my_model.h5'
    model = tf.keras.models.load_model(model_path)
    
    # class 지정
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']

    cap = cv2.VideoCapture(0)  # 카메라 인덱스 (일반적으로 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 마스크 색상 범위를 정의합니다. 살구색의 경우, 다음과 같이 범위를 설정할 수 있습니다.
        lower_bound = np.array([0, 20, 70])  # 하한값
        upper_bound = np.array([20, 255, 255])  # 상한값

        # 마스크를 생성합니다.
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 원본 이미지에 마스크를 적용하여 살구색만 남깁니다.
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        def preprocess(image):
            resized_image = cv2.resize(image, (128, 128))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            normalized_image = gray_image / 255.0
            reshaped_image = np.reshape(normalized_image, (1, 128, 128, 1))
            return reshaped_image

        processed_frame = preprocess(blurred_frame)

        predictions = model.predict(processed_frame)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]

        text = f"Predicted Class: {predicted_class_label}"
        cv2.putText(masked_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(predicted_class_label)
        image = Image.fromarray(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image)
        
        # video_label을 업데이트하는 함수 호출
        update_video_label(img)
        
        # z를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tot_text.append(predicted_class_label)
    # 문자열(메소드)를 하나의 텍스트로 변환
    encText = ''.join(tot_text)
    label3.config(text=encText)  # 단어 출력 위치

def update_video_label(img):
    video_label.configure(image=img)
    video_label.image = img
    
def signlanguage():
    global predicted_class_label
    global tot_text
    global encText
   

    # 프레임 처리를 담당하는 함수를 쓰레드로 실행
    threading.Thread(target=process_frame).start()
    
def Output(): #단어 출력
    
    #문자열(메소드)를 하나의 텍스트로 변환
    encText = ''.join(tot_text)
    label3.config(text=encText) #단어 출력 위치
    
def translation():
    import os
    import sys
    import urllib.request
    import urllib.parse
    import json

    encText = ''.join(tot_text)

    print(encText)

    client_id = "SGnoCu207DwIN6PKhSwo"
    client_secret = "lRUBh6ujXI"
    encText = urllib.parse.quote(encText)
    # 입력언어 영어, 출력언어 한국어 변환
    data = "source=en&target=ko&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read().decode('utf-8')
        parsed_response = json.loads(response_body)
        translated_text = parsed_response['message']['result']['translatedText']
        # 번역된 텍스트 출력
        print(translated_text)
        label3.config(text=translated_text)
    else:
        print("Error Code:" + rescode)

   
    
def Reset():
    global tot_text
    global encText
    tot_text = []
    encText = ''
    label3.config(text=encText)
    

def spacing():
    global tot_text
    global encText
    tot_text.append(" ")  # 띄어쓰기를 문자열 리스트에 추가
    encText = ''.join(tot_text)  # 문자열 리스트를 하나의 문자열로 변환

btn = Button(frame4,  text='수어인식', command=signlanguage,font = ("궁서체",15))

btn_3 = Button(frame4,  text='띄어쓰기', command=spacing,font = ("궁서체",15))
btn_4 = Button(frame4,  text='초기화', command=Reset,font = ("궁서체",15))
btn_5 = Button(frame4,  text='번역', command=translation,font = ("궁서체",15))

btn.config(width=20, height=5) 
btn_3.config(width=20, height=5) 
btn_4.config(width=20, height=5) 
btn_5.config(width=20, height=5)

btn.place(x=75, y=139)
btn_3.place(x=75, y=255)
btn_4.place(x=75, y=372)
btn_5.place(x=75, y=488)




label1 = Label(frame6,text= "수어 번역기", font = ("궁서체",50))
label2 = Label(frame4,text= "입력된 텍스트", font = ("궁서체",40))
label3 = Label(frame4,text= "", font = ("궁서체",20)) # 텍스트 출력 라벨

btnToFrame4 = Button(frame6,text="음성 인식",padx=24,pady=10,command=lambda:[openFrame(frame6)],)
btnToFrame6 = Button(frame6,text="수어 번역",padx=24,pady=10,command=lambda:[openFrame(frame4)],font = ("궁서체",15))


label1.pack(padx=0,pady=0)
label2.pack(padx=20,pady=20)
label3.pack(padx=60,pady=40)
btnToFrame4.pack()
btnToFrame6.pack()


    
window.mainloop()