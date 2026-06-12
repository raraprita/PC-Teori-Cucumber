from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best.pt")

disease_info = {
    "Anthracnose": {
        "desc": "Penyakit jamur yang menyebabkan bercak coklat kehitaman pada daun dan buah.",
        "solution": "Gunakan fungisida dan buang bagian tanaman yang terinfeksi."
    },
    "Bacterial-Wilt": {
        "desc": "Penyakit layu akibat infeksi bakteri.",
        "solution": "Perbaiki drainase dan cabut tanaman yang terinfeksi."
    },
    "Belly-Rot": {
        "desc": "Busuk pada bagian buah yang menyentuh tanah.",
        "solution": "Gunakan mulsa dan hindari kontak langsung buah dengan tanah."
    },
    "Downy-Mildew": {
        "desc": "Penyakit jamur yang menyebabkan bercak kuning pada daun.",
        "solution": "Kurangi kelembapan lahan dan gunakan fungisida."
    },
    "Fresh-Leaf": {
        "desc": "Daun timun dalam kondisi sehat.",
        "solution": "Tidak diperlukan tindakan khusus."
    },
    "Fresh-Cucumber": {
        "desc": "Buah timun dalam kondisi sehat.",
        "solution": "Tidak diperlukan tindakan khusus."
    },
    "Gummy-Stem-Blight": {
        "desc": "Penyakit yang menyebabkan batang mengeluarkan lendir dan mengering.",
        "solution": "Lakukan sanitasi lahan dan gunakan fungisida."
    },
    "Pythium-Fruit-Rot": {
        "desc": "Busuk buah akibat infeksi jamur Pythium.",
        "solution": "Kurangi kelembapan dan lakukan rotasi tanaman."
    }
}


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    description = None
    solution = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = str(uuid.uuid4()) + ".jpg"

            upload_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(upload_path)

            results = model(
                upload_path,
                conf=0.05
            )

            result = results[0]

            if len(result.boxes) > 0:

                names = result.names

                cls_id = int(
                    result.boxes.cls[0]
                )

                confidence = float(
                    result.boxes.conf[0]
                )

                prediction = names[cls_id]

                if prediction in disease_info:

                    description = disease_info[prediction]["desc"]

                    solution = disease_info[prediction]["solution"]

            else:

                prediction = "Tidak terdeteksi"

                description = "Model tidak menemukan penyakit pada gambar."

                solution = "Coba gunakan gambar yang lebih jelas atau dekat."

            result_filename = (
                "result_" + filename
            )

            result_path = os.path.join(
                RESULT_FOLDER,
                result_filename
            )

            result.save(
                filename=result_path
            )

            image_path = result_path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description,
        solution=solution,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )