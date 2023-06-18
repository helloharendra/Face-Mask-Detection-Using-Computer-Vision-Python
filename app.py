from PIL import Image as Im
import os
import webcam
import streamlit as st
from configu import *
from detect_mask_image import detect
from db import Video,Image
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import cv2
# from converter import convert_video
from detect_mask_video import video

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_image(file,path):
    try:
        db = opendb()
        file =  os.path.basename(path)
        name, ext = file.split('.') # second piece
        img = Image(filename=name,extension=ext,filepath=path)
        db.add(img)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(DONE_BY)

choice = st.sidebar.radio("select option",MENU_OPTION)

if choice == 'About project':
    st.title("About Our Project")
    st.image('image/maskgirls.png')
    st.info('''Face Mask Detection Platform utilizes Artificial Network to perceive if a person does/doesn't wear a mask.
The application can be associated with any currentor new cameras to identify individuals with/without a mask.
By developing a face mask detection technique,
through which we identify people weared mask or not.
so we suggest people who not weared mask to wear  it to reduce the covid/ any viral disease spread''')
    


if choice == 'Instruction to use':
    st.title("HOW TO USE THE APPLICATION")
    st.info(''' step 1. install the software on your computer.\n''')
    st.image('image/Softwareinstall.jpeg')
    st.info('''step 2. Connect your computer with webcam.\n''')
    st.image('image/webcamset.jpeg')
    st.info('''step 3. click on the realtime detection if you want detection through live stream.\n''')
    st.image('image/realtime.png')
    st.info('''step 4. Now click on "start camera window" button to start detection.\n''')
    st.image('image/instruction_realtime.PNG')
    st.info('''->Here you go.......\n''')
    st.image('image/9.PNG')
    st.info('''step 5.click on image based test if you want to detect through providing images.\n''')
    st.info('''->n the right hand side now you see "Browse File" button to upload image.\n''')
    st.image('image/instruction_imgbased.PNG')
    st.info('''->Now select your image for detection of mask.\n''')
    
    st.image('image/instruction_imgbasedbrowse.PNG')
    st.info('''->Here you go.......\n''')
    st.image('image/instruction_imgbasedresult.png')
    st.image('image/instruction_imgbasedresultmasked.PNG')
    
    


if choice == 'Sample dataset':
    st.title("Dataset Samples")
    st.image('sampleImage/b.jpeg')
    st.image('sampleImage/i.jpeg')
    st.image('sampleImage/c.jpeg')
    st.image('sampleImage/d.jpeg')
    st.image('sampleImage/e.jpeg')
    st.image('sampleImage/g.jpeg')
    st.image('sampleImage/h.jpeg')
    st.image('sampleImage/a.jpeg')
    st.image('sampleImage/j.jpeg')
    st.image('sampleImage/f.jpeg')


if choice == 'Camera based test':
   st.title("Real time camera based test")
   btn = st.button('start realtime AI camera')
   if btn:
       webcam.load_camera(num=0)


if choice == 'image based test':
    st.title("Upload images for image based test")
    st.subheader('select an image')
    img = st.file_uploader("browse to select",type=['jpg','png','jpeg'])
    if img:
        im = Im.open(img)
        # create a address for image path
        path = os.path.join("images",img.name)
        # save file to upload folder
        im.save(path,format=img.type.split('/')[1])
        status = save_image(img,path)
        if status:
            st.sidebar.success("file uploaded")
            col1 ,col2 = st.beta_columns(2)
            col1.image(path,use_column_width=True,caption='original')
            out_img = detect(path)
            cv2.imwrite(path,out_img)
            col2.image(path,use_column_width=True,caption='prediction')
        else:
            st.sidebar.error('upload failed')

if choice  =='realtime detection':
    st.title("Real time camera based test")
    
    cnf = st.slider('confidence threshold',min_value=.1, max_value=1.0,value=.5)
    btn = st.button("start camera window")
    st.info('Click on start camer window for detection ')
    if btn:
        video(cnf=cnf)
if choice == 'view previous predictions':
    db = opendb()
    results = db.query(Image).all()
    db.close()        