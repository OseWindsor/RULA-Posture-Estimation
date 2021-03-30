from flask import Flask, request, render_template
import os
from os import path
from pathlib import Path
from rula_helper_functions import plot_estimate_rula_score
import time
import cv2

app = Flask(__name__)

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'

rula_result_dict = {}
def upload_file(side_img, front_img):
    if side_img != '' and front_img !='':
        img_name = side_img.split('\\')[-1]
        side_img_path = Path(os.path.join('static/uploads',img_name))
        print('saving image in path: ',side_img_path)
        cv2.imwrite(img_name,side_img_path)

        img_name = front_img.split('\\')[-1]
        front_img_path = Path(os.path.join('static/uploads', img_name))
        print('saving image in path: ', front_img_path)
        cv2.imwrite(img_name, front_img_path)


@app.route("/")
def home():
    return render_template("rula_app.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict_keypoints():
    rula_result_dict_main = {}
    uploaded_img_s = request.files['side_img']
    uploaded_img_f = request.files['front_img']
    if len(request.form)!=0:
        is_posture_lean = True if request.form.getlist('posture_lean')[0] == 'on' else False
        is_shoulder_raise = True if request.form.getlist('shoulder_raise')[0] == 'on' else False
    else:
        is_posture_lean = False
        is_shoulder_raise = False
    user_param_list = [is_posture_lean, 'R', is_shoulder_raise]
    if uploaded_img_s.filename != '' and uploaded_img_f.filename != '':
        uploaded_side_img_path = 'static/uploads/'+uploaded_img_s.filename
        uploaded_front_img_path = 'static/uploads/'+uploaded_img_f.filename
        uploaded_img_s.save(uploaded_side_img_path)
        uploaded_img_f.save(uploaded_front_img_path)
        rula_result_dict_main = plot_estimate_rula_score(input_test_side_img=uploaded_side_img_path,
                                                         input_test_front_img=uploaded_front_img_path,
                                                         user_param = user_param_list)
    return render_template("result.html",rula_result_dict = rula_result_dict_main)

@app.route("/predict_postures", methods=['GET', 'POST'])
def predict_multiple_keypoints():
    rula_result_dict_main = {}
    if len(request.form)!=0:
        is_posture_lean = True if request.form.getlist('multiple_posture_lean')[0] == 'on' else False
        is_shoulder_raise = True if request.form.getlist('multiple_shoulder_raise')[0] == 'on' else False
    else:
        is_posture_lean = False
        is_shoulder_raise = False
    user_param_list = [is_posture_lean, 'R', is_shoulder_raise]
    front_img = 'static/uploads/example/DSC_0006_F.JPG'
    side_img = 'static/uploads/example/DSC_0005_R.JPG'
    uploaded_posture_imgs = request.files.getlist('posture_img_folder')
    print('No of images to be uploaded: ', len(uploaded_posture_imgs))
    # Create Folders
    for uploaded_posture_img in uploaded_posture_imgs:
        if uploaded_posture_img.filename != '' :
            print('File name: ',uploaded_posture_img.filename)
            folder_list_names = str(uploaded_posture_img.filename).rsplit(sep='/', maxsplit=1)
            if len(folder_list_names)>0:
                for folder_list_name in folder_list_names:
                    if any(img_extensions in folder_list_name for img_extensions in ['.jpg','.JPG','.png','.PNG','.jpeg']):
                       print('Its a file')
                    else:
                        upload_dir = 'static/uploads/'+folder_list_name
                        if not path.exists(upload_dir):
                            print('Create a folder: ',upload_dir)
                            os.makedirs(upload_dir)
    # Save files
    for uploaded_posture_img in uploaded_posture_imgs:
        if uploaded_posture_img.filename != '':
            upload_img = 'static/uploads/'+ uploaded_posture_img.filename
            if not path.exists(upload_img):
                print('Uploading File name: ', uploaded_posture_img.filename)
                uploaded_posture_img.save(upload_img)
    # Traverse through each directory
    parent_upload_folder = Path(os.path.join('static/uploads/',str(uploaded_posture_imgs[0].filename).split(sep='/', maxsplit=1)[0]))
    test_img_folder_list = [os.path.join(parent_upload_folder, f) for f in os.listdir(parent_upload_folder) if
                            not os.path.isfile(os.path.join(parent_upload_folder, f))]
    print('No of folders in test image directory: ' + str(len(test_img_folder_list)))
    for test_img_folder in test_img_folder_list:
        img_view = 'R'
        dict_key = test_img_folder
        test_img_path_list = [os.path.join(test_img_folder, f) for f in os.listdir(test_img_folder) if
                              os.path.isfile(os.path.join(test_img_folder, f))]
        for i in range(2):

            img_view_flag = test_img_path_list[i].rsplit(sep='.', maxsplit=1)[0].rsplit(sep='_', maxsplit=1)[1]
            if img_view_flag == 'F':
                front_img = test_img_path_list[i]
            else:
                img_view = img_view_flag
                side_img = test_img_path_list[i]
        if os.path.isfile(front_img) and os.path.isfile(side_img):
            start_time_1 = time.time()
            rula_result_dict = plot_estimate_rula_score(front_img, side_img, user_param = user_param_list)
            elapsed_time_1 = time.time() - start_time_1
            print('Time taken to inference posture: ' + str(elapsed_time_1) + ' seconds')
            rula_result_dict_main[dict_key] = rula_result_dict


    return render_template("multiple_posture_result.html",rula_multiple_posture_dict = rula_result_dict_main)

if __name__ == "__main__":
    app.run(debug=True)