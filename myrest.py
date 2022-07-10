import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import mysearch as S
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def upload_file():
    # S.modelLoad()
    return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file3():
   if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        S.modelTest(f.filename)
        hists = os.listdir('static/result')
        result = ['result/' + file for file in hists]
        return render_template('result.html', result=result)
        
if __name__ == '__main__':
   app.run(debug = True)