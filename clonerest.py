import imghdr
from flask import Flask, render_template, jsonify ,request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
import os
import clonesearch as S
import cloneclasify as C
import caseclassify as CC
# import ResnetVit_brain_test_prediction as VV
import unet_segmentation_hem27case as Seg
import heatmap as H
import json
import glob
from torch import Tensor


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10240 * 10240
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = './uploads'
app.config['CASE_PATH'] = './static/case'
app.config['QUERY_PATH'] = './static/query'
app.config['VIT_PATH'] = './static/vit/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/searchResult', methods=['POST','GET'])
def searchUpload():
    print("searchupload start")
    uploaded_file = request.files['file']
    print("One")
    filename = uploaded_file.filename
    file_path = os.path.join(app.config['UPLOAD_PATH'],filename)
    uploaded_file.save(file_path)
    status = S.call_search(file_path)
    result_url_json = json.dumps(status)
    results = os.listdir('./static/results')
    return render_template('results.html', results = results, filename=filename)

@app.route('/clasifyResult', methods=['POST', 'GET'])
def clasifyUpload():
    print("clasifyupload start")
    uploaded_file = request.files['file']
    print("Two")
    filename = uploaded_file.filename
    file_path = os.path.join(app.config['UPLOAD_PATH'],filename)
    uploaded_file.save(file_path)
    # C.modelLoad(file_path)
    pred = H.getheatmap(file_path)
    # results = os.listdir('./static/results')
    heatmap = os.listdir('./static/heatmap')
    # print(results)
    return render_template('results.html', heatmap=heatmap, pred=pred)

@app.route('/caseclassifyupload', methods=['POST', 'GET'])
def caseclasifyUpload():
    print("case clasify upload start")
    uploaded_files = request.files.getlist("file[]")
    actual_case = request.form.get('case')
    file_list = []
    file_name = []
    results = os.listdir('static/case/')
    for f in results:
        os.remove(os.path.join('static/case/', f))
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['CASE_PATH'], filename)
        file_list.append(file_path)
        file_name.append(filename)
        file.save(file_path)
    file_list.sort()
    file_name.sort()
    print(file_name, file_list)
    prediction = CC.getCaseClassify(file_list)
    correct_pred = 0
    wrong_pred = 0
    Ham_Count = 0
    Inf_Count = 0
    Norm_Count = 0
    for x in prediction:
        if actual_case == x:
            correct_pred = correct_pred + 1
        else:
            wrong_pred = wrong_pred + 1
        if x=="Haemorrhage":
            Ham_Count = Ham_Count + 1
        if x=="Infarct":
            Inf_Count = Inf_Count + 1
        if x=="Normal":
            Norm_Count = Norm_Count + 1
    case_images = os.listdir('./static/case')
    case_images.sort()
    return render_template('caseclassifyupload.html', file_name=file_name, len=len(case_images), image=case_images, pred=prediction, actual_case=actual_case, correct_pred=correct_pred, wrong_pred=wrong_pred, Ham_Count=Ham_Count, Inf_Count=Inf_Count, Norm_Count=Norm_Count)

@app.route('/vitclassifyupload', methods=['POST', 'GET'])
def vitclasifyUpload():
    print("vit clasify upload start")
    uploaded_files = request.files.getlist("file[]")
    actual_case = request.form.get('case')
    file_list = []

    results = os.listdir('static/vit/Haemorrhage')
    for f in results:
        os.remove(os.path.join('static/vit/Haemorrhage', f))

    results = os.listdir('static/vit/Infarct')
    for f in results:
        os.remove(os.path.join('static/vit/Infarct', f))

    results = os.listdir('static/vit/Normal')
    for f in results:
        os.remove(os.path.join('static/vit/Normal', f))

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['VIT_PATH']+actual_case, filename)
        file_list.append(file_path)
        file.save(file_path)
    
    print("File Uploaded......")
    prediction = VV.getVitClassify()
    # import ResnetVit_brain_test_prediction as VV
    # file = open(r'/home/user/Documents/webapp/ResnetVit_brain_test_prediction.py', 'rb').read()
    # exec(file)
    correct_pred = 0
    prediction = []
    wrong_pred = 0
    for x in prediction:
        if actual_case == x:
            correct_pred = correct_pred + 1
        else:
            wrong_pred = wrong_pred + 1
    case_images = os.listdir('./static/vit/' + actual_case)
    return render_template('vitclassifyupload.html', len=len(case_images), image=case_images, pred=prediction, actual_case=actual_case, correct_pred=correct_pred, wrong_pred=wrong_pred)
    #return render_template('vitclassifyupload.html', len=len(case_images), image=case_images, actual_case=actual_case, correct_pred=correct_pred, wrong_pred=wrong_pred)

@app.route('/segmentationupload', methods=['POST', 'GET'])
def segmentationupload():
    print("segmentation upload start")
    uploaded_files = request.files.getlist("file[]")
    actual_case = request.form.get('case')
    file_list = []

    results = os.listdir('static/segmentation/input')
    for f in results:
        os.remove(os.path.join('static/segmentation/input', f))

    results = os.listdir('static/segmentation/output')
    for f in results:
        os.remove(os.path.join('static/segmentation/output', f))
        
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join("static/segmentation/input/", filename)
        file_list.append(filename)
        file.save(file_path)
    
    print("File Uploaded......")
    print(file_list)
    Seg.segmentationFun(file_list)
    input_images = sorted(os.listdir('./static/segmentation/input/'))
    output_images = sorted(os.listdir('./static/segmentation/output/'))
    print(input_images)
    print(output_images)
    return render_template('segmentationupload.html', len=len(input_images), input_images=input_images, output_images=output_images)

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/classify')
def clasify():
    return render_template('classify.html')

@app.route('/caseclassify')
def caseclassify():
    return render_template('caseclassify.html')

@app.route('/vitclassify')
def vitclassify():
    return render_template('vitclassify.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

