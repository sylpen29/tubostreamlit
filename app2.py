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
from streamlit_webrtc import WebRtcMode, VideoHTMLAttributes, webrtc_streamer, VideoProcessorBase


# def get_gpu_memory():
#     result = subprocess.check_output(
#         [
#             'nvidia-smi', '--query-gpu=memory.used',
#             '--format=csv,nounits,noheader'
#         ], encoding='utf-8')
#     gpu_memory = [int(x) for x in result.strip().split('\n')]
#     return gpu_memory[0]

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
# path_model_file = st.sidebar.text_input(
#     'path to YOLOv7 Model:',
#     'best.pt', label_visibility="hidden"
# )
path_model_file = "best.pt"

                                                                    # Class txt
# path_to_class_txt = st.sidebar.file_uploader(
#         'class.txt:', type=['txt'], label_visibility="hidden"

# )
with open('class.txt', 'r') as file:
          lines = file.read()
path_to_class_txt = io.StringIO(lines)
                                                                # read class.txt
# bytes_data = path_to_class_txt.getvalue()
# class_labels = bytes_data.decode('utf-8').split("\n")
# color_pick_list = []
bytes_data = path_to_class_txt.getvalue()
class_labels = bytes_data.split("\n")
color_pick_list = []

for i in range(len(class_labels)):
    classname = class_labels[i]
    color = color_picker_fn(classname, i)
    color_pick_list.append(color)


if path_to_class_txt is not None:

    # options = st.sidebar.radio(
    #     'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)
    # options = 'Webcam'
    # gpu_option = st.sidebar.radio(
    #     'PU Options:', ('CPU', 'GPU'))

    # if not torch.cuda.is_available():
    #     st.sidebar.warning('ready!', icon="🚨")
    # else:
    #     st.sidebar.success(
    #         'ready!',
    #         icon="✅"
    #     )

    # Confidence
    confidence = st.sidebar.slider(
        'indice de confiance', min_value=0.0, max_value=1.0, value=0.75)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'épaisseur du trait:', min_value=1,
        max_value=5, value=2
    )


    model = custom(path_or_model=path_model_file)

    
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        bbox_list = []
        current_no_class = []
        results = model(frm)

        stframe2 = st.empty()

        
                    # Bounding Box
        box = results.pandas().xyxy[0]
        class_list = box['class'].to_list()

        for i in box.index:
                xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                    int(box['ymax'][i]), box['confidence'][i]
                if conf > confidence:
                    bbox_list.append([xmin, ymin, xmax, ymax])
        if len(bbox_list) != 0:
                for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, frm, label=class_labels[id],
                                    color=color_pick_list[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])
        
        # FPS
        # c_time = time.time()
        # fps = 1 / (c_time - p_time)
        # p_time = c_time

                                   # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Forme', 'Quantité'])
        
       
        with stframe2.container():
            st.markdown("<h3>Nombre de tubes</h3>", unsafe_allow_html=True)
            st.dataframe(df_fq, use_container_width=True)
        
        #FRAME_WINDOW.image(frm, channels='BGR')
        return av.VideoFrame.from_ndarray(frm, format='bgr24')
        
    


muted = st.checkbox("Mute")

webrtc_streamer( key="Tubocomptage", 
                video_processor_factory=VideoProcessor, 
                video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}, muted=muted ))    
   