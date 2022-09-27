from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from werkzeug.utils import secure_filename
from facial_mask_game import Run
from PIL import Image

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
CORS(app)

def upload(animal):
    f = request.files.get('file')
    im = Image.open(f)
    filename = secure_filename(f.filename)

    print(filename)

    random_num = random.randint(0, 100)
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + filename.rsplit('.', 1)[1]
    file_path = basedir + "/static/file/"

    if not os.path.exists(file_path):
        os.makedirs(file_path, 755)

    im.save(file_path + filename)
    # os.remove(file_path + filename)
    result = Run(im, animal)
    result.save(file_path +"mask" +filename)

    #change my_host to the ip that frontend can connect with
    my_host = "http://137.189.94.89:5000"
    # my_host = "http://circlema.ddns.net:9999"
    mask_path_file = my_host + "/static/file/" +"mask"+ filename
    data = {"msg": "success", "maskUrl":mask_path_file}

    payload = jsonify(data)
    return payload,200

@app.route("/upload/cat", methods=["POST"])
def uploadcat():
    return upload("cat")

@app.route("/upload/tiger", methods=["POST"])
def uploadtiger():
    return upload("tiger")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port="5000")