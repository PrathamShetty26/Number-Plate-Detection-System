from flask import Flask, render_template, request
from anpr import read_img, read_video
import os

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, '/Users/Lenovo/Desktop/new/upload')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        upload_video = None
        upload_img = None

        for file in request.files:
            filename = request.files[file].filename
            extension = os.path.splitext(filename)[1].lower()

            if extension in ['.mp4', '.mov']:
                upload_video = request.files[file]
            elif extension in ['.jpg', '.jpeg', '.png']:
                upload_img = request.files[file]

        if (upload_video != None):
            filename = upload_video.filename
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_video.save(path_save)
            text = read_video(path_save)
            print(text)
            return render_template('index.html', upload=True, text=text)

        if (upload_img != None):
            filename = upload_img.filename
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_img.save(path_save)
            text = read_img(path_save)
            print(text)
            return render_template('index.html', upload=True, text=text)

    return render_template('index.html', upload=False, text='')


if __name__ == "__main__":
    app.run(debug=True)
