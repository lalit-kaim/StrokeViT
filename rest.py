
# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask <-- study material
# https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

#convert the request and responce in json
#responce inlcude by default variable status success or not success 
# describe 200 404 not success 

import imghdr
from flask import Flask, render_template, jsonify ,request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
import os
import search as S
import json
#file path or url

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = 'uploads'

# def validate_image(stream):
#     header = stream.read(512)  # 512 bytes should be enough for a header check
#     stream.seek(0)  # reset stream pointer
#     format = imghdr.what(None, header)
#     if not format:
#         return None
#     return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files )

@app.route('/', methods=['POST','GET'])
def upload_files():

    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    file_path = os.path.join(app.config['UPLOAD_PATH'],filename)
    uploaded_file.save(file_path)
    status = S.call_search(file_path)

    print (status)
    result_url_json = json.dumps(status)

    results = os.listdir('static/results')
    # print(results)
    return render_template('results.html', results = results)

    # return redirect(url_for('index'))

# @app.route('/uploads/<filename>',methods=['GET'])
# def upload(filename):

#     return send_from_directory(app.config['UPLOAD_PATH'], filename)

# @app.route('/results/', methods=['GET'])
# def results():
#     results = os.listdir('static/results')
#     # print(results)
#     return render_template('results.html', results = results)

if __name__ == '__main__':

    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

