import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
import io
import av
import threading
from streamlit_webrtc import WebRtcMode, VideoHTMLAttributes, webrtc_streamer



# def color_picker_fn(classname, key):
#     color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
#     color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
#     color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
#     return color
def color_picker_fn(classname, key):
    if classname == "rond":
        color = "#ff0003"
    elif classname == "carre":
        color = "#00ff1b"
    elif classname == "rect":
        color = "#0059ff"
    elif classname == "H":
        color = "#ffe000"
    color_picke = st.sidebar.color_picker(f'{classname}', color , key=key)    
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color


p_time = 0

st.title('Tubocomptage')
sample_img = cv2.imread('logo_tubo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('réglages')

                                                                # path to model

path_model_file = "best.pt"

                                                                    # Class txt

with open('class.txt', 'r') as file:
          lines = file.read()
path_to_class_txt = io.StringIO(lines)





                                                                # read class.txt

bytes_data = path_to_class_txt.getvalue()
class_labels = bytes_data.split("\n")
color_pick_list = []

for i in range(len(class_labels)):
    classname = class_labels[i]
    color = color_picker_fn(classname, i)
    color_pick_list.append(color)



if path_to_class_txt is not None:

   
    # Confidence
    confidence = st.sidebar.slider(
        'indice de confiance', min_value=0.0, max_value=1.0, value=0.55)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'épaisseur du trait:', min_value=1,
        max_value=5, value=2
    )


    model = custom(path_or_model=path_model_file)

    
class VideoProcessor:

    def __init__(self):
        pass


      
    def recv(self, frame, bbox_list, current_no_class):
        self.frame = frame
        self.bbox_list = bbox_list
        self.current_no_class = current_no_class
        
        
        self.frame = frame.to_ndarray(format="bgr24")
        self.bbox_list = []
        self.current_no_class = []  
        results = model(self.frame)

        

        
                    # Bounding Box
    
        box = results.pandas().xyxy[0]
        class_list = box['class'].to_list()

        for i in box.index:
                xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                    int(box['ymax'][i]), box['confidence'][i]
                if conf > confidence:
                    self.bbox_list.append([xmin, ymin, xmax, ymax])
        if len(self.bbox_list) != 0:
                for bbox, id in zip(self.bbox_list, class_list):
                        plot_one_box(bbox, self.frame, label=class_labels[id],
                                    color=color_pick_list[id], line_thickness=draw_thick)
                        self.current_no_class.append([class_labels[id]])


    def compter(self):

        stframe2 = st.empty()                                  
        class_fq = dict(Counter(i for sub in self.current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Forme', 'Quantité'])
                
            
        with stframe2.container():
            st.markdown("<h3>Nombre de tubes</h3>", unsafe_allow_html=True)
            st.dataframe(df_fq, use_container_width=True)                        
        
        return av.VideoFrame.from_ndarray(self.frame, format='bgr24')    
  

                                # Current number of classes

     
           
    
webrtc_ctx = webrtc_streamer( key="Tubocomptage",
                mode=WebRtcMode.SENDRECV, 
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_processor_factory=VideoProcessor,
                video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"} )) 





   





              