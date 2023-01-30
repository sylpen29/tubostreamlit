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


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]


# liste des labels dans la sidebar avec changement de couleur
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

# barre d'affichage sur le côté gauche
st.title('Tubocomptage')
sample_img = cv2.imread('logo_tubo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('réglages')

# path du model

path_model_file = "best.pt"

# On récupère les labels "Class.txt"

with open('class.txt', 'r') as file:
          lines = file.read()
path_to_class_txt = io.StringIO(lines)



if path_to_class_txt is not None:

    options = 'Webcam'
    
    gpu_option = 'CPU'

    # if options == 'Webcam':
    #      cam_options = st.sidebar.selectbox('Webcam Channel',
    #                                          ('Select Channel', '0', '1', '2', '3'))
    cam_options = '0'
    # if not torch.cuda.is_available():
    #     st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
    # else:
    #     st.sidebar.success(
    #         'Succès! Détection en cours...',
    #         icon="✅"
    #     )

    # Confiance
    confidence = st.sidebar.slider(
        'Indice de confiance', min_value=0.0, max_value=1.0, value=0.55)

    # épaisseur du trait des box
    draw_thick = st.sidebar.slider(
        'Epaisseur du trait:', min_value=1,
        max_value=20, value=2
    )
    
    # lecture des labels "class.txt"
    
    bytes_data = path_to_class_txt.getvalue()
    class_labels = bytes_data.split("\n")
    color_pick_list = []

    for i in range(len(class_labels)):
        classname = class_labels[i]
        color = color_picker_fn(classname, i)
        color_pick_list.append(color)

    # # Image
    # if options == 'Image':
    #     upload_img_file = st.sidebar.file_uploader(
    #         'Upload Image', type=['jpg', 'jpeg', 'png'])
    #     if upload_img_file is not None:
    #         pred = st.checkbox('Predict Using YOLOv7')
    #         file_bytes = np.asarray(
    #             bytearray(upload_img_file.read()), dtype=np.uint8)
    #         img = cv2.imdecode(file_bytes, 1)
    #         FRAME_WINDOW.image(img, channels='BGR')

    #         if pred:
    #             if gpu_option == 'CPU':
    #                 model = custom(path_or_model=path_model_file)
    #             if gpu_option == 'GPU':
    #                 model = custom(path_or_model=path_model_file, gpu=True)
                
    #             bbox_list = []
    #             current_no_class = []
    #             results = model(img)
                
    #             # Bounding Box
    #             box = results.pandas().xyxy[0]
    #             class_list = box['class'].to_list()

    #             for i in box.index:
    #                 xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
    #                     int(box['ymax'][i]), box['confidence'][i]
    #                 if conf > confidence:
    #                     bbox_list.append([xmin, ymin, xmax, ymax])
    #             if len(bbox_list) != 0:
    #                 for bbox, id in zip(bbox_list, class_list):
    #                     plot_one_box(bbox, img, label=class_labels[id],
    #                                  color=color_pick_list[id], line_thickness=draw_thick)
    #                     current_no_class.append([class_labels[id]])
    #             FRAME_WINDOW.image(img, channels='BGR')


    #             # Current number of classes
    #             class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
    #             class_fq = json.dumps(class_fq, indent = 4)
    #             class_fq = json.loads(class_fq)
    #             df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                
    #             # Updating Inference results
    #             with st.container():
    #                 st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
    #                 st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
    #                 st.dataframe(df_fq, use_container_width=True)

    # Web-cam detection
    
    
    # Model
    model = custom(path_or_model=path_model_file, gpu=False)


    if len(cam_options) != 0:
        if not cam_options == 'Select Channel':
            cap = cv2.VideoCapture(int(cam_options))
            stframe1 = st.empty()
            stframe2 = st.empty()
            # stframe3 = st.empty()
            while True:
                success, img = cap.read()
                if not success:
                    st.error(
                        f'Webcam channel {cam_options} NOT working\n \
                        Change channel or Connect webcam properly!!',
                        icon="🚨"
                    )
                    break

                bbox_list = []
                current_no_class = []
                results = model(img)
                    
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
                        plot_one_box(bbox, img, label=class_labels[id],
                            color=color_pick_list[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])
                FRAME_WINDOW.image(img, channels='BGR')

                    # FPS
                c_time = time.time()
                fps = 1 / (c_time - p_time)
                p_time = c_time
                    
                    # Current number of classes
                class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent = 4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=['Forme', 'Quantité'])
                   
                    
                    # Updating Inference results
                with stframe1.container():
                    st.markdown("<h5>Inférences</h5>", unsafe_allow_html=True)
                    if round(fps, 4)>1:
                        st.markdown(f"<h6 style='color:green;'>Fps: {round(fps, 4)}</h6>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h6 style='color:red;'>Fps: {round(fps, 4)}</h6>", unsafe_allow_html=True)

                with stframe2.container():
                    st.markdown("<h3>Nombre de tubes par forme</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq, use_container_width=True)
