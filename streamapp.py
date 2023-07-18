import base64
from ultralytics import YOLO
import cv2
import math
import streamlit as st
import time

header_style = """
<style>
h1 {
    font-size: 32px !important;
    margin-bottom: 20;
}
</style>
"""

st.set_page_config(page_title='VW: Defect Detection',
                   page_icon='🔧', layout='wide',)


image_path = 'VWlogo.png'

st.markdown(header_style, unsafe_allow_html=True)
img_col, mid_col, tit_col = st.columns((1, 0.5, 20))
img_col.image(image_path, width=100)
tit_col.title('VWITS Defect Detection')
run = st.checkbox('Run')
video_capture = 0
# Create a Webcam Object
cap = cv2.VideoCapture(video_capture)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
st.write("")
st.write("")
st.write("")
st.write("")
left_column, right_column = st.columns((0.6, 0.3))
with left_column:
    FRAME_WINDOW = st.image([])
with right_column:
    title = st.empty()
    bottle_counter = st.empty()
    cap_counter = st.empty()
    label_counter = st.empty()
    defbottle_counter = st.empty()
    counter = st.empty()

model = YOLO("ppe.pt")
classNames = ['Bottle', 'Cap', 'Defective Bottle', 'Label']
# thresh = st.slider('Set Confidence Threshold', 0, 100, 25)
while run:
    with title.container():
        st.header("Detections")
        st.divider()
        st.write("")
    with bottle_counter.container():  
        st.info(
            f"1. Normal Bottle Detected:&emsp;&emsp;&emsp;&emsp;&emsp;No")
        st.write("")
        st.write("")
        st.write("")
    with label_counter.container():
        st.info(
            f"3. Label Detected:     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;   No")
        st.write("")
        st.write("")
        st.write("")
    with cap_counter.container():
        st.info(
            f"2. Cap Detected: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;No")
        st.write("")
        st.write("")
        st.write("")
    with defbottle_counter.container():
        st.info(
            f"4. Defective Bottle Detected:  &emsp;&emsp;&emsp;&emsp;No")
        st.write("")
        st.write("")
        st.write("")
    # with counter.container():
    #     st.write(
    #         f"<span style='font-size: 40px; color: red;'>5. Nothing Detected:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;No</span>", unsafe_allow_html=True)
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]    
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            if class_name == 'Bottle':
                color = (0, 204, 255)
                with bottle_counter.container():
                    st.success(
                        f"1. Normal Bottle Detected:&emsp;&emsp;&emsp;&emsp;&emsp;Yes")
                    st.metric(label=class_name,value=conf)
            elif class_name == "Label":
                color = (222, 82, 175)
                with label_counter.container():
                    st.success(
                        f"2. Label Detected:     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;   Yes")
                    st.metric(label=class_name,value=conf)
            elif class_name == "Cap":
                color = (222, 82, 175)
                with cap_counter.container():
                    st.success(
                        f"3. Cap Detected: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Yes")
                    st.metric(label=class_name,value=conf)
            elif class_name == "Defective Bottle":
                color = (0, 149, 255)
                with defbottle_counter.container():
                    st.error(
                        f"4. Defective Bottle Detected:  &emsp;&emsp;&emsp;&emsp;Yes", icon="🚨")
                    st.metric(label=class_name,value=conf)
            # elif class_name == "":
            #     color = (85, 45, 255)
            #     with counter.container():
            #         st.write(
            #             f"<span style='font-size: 40px; color: green;'>5. No Detections:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Yes</span>", unsafe_allow_html=True)
            
            if conf > 0.6:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -
                              1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1-2), 0, 1,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            time.sleep(1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1500, 900))
    FRAME_WINDOW.image(frame)