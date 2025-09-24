import uuid
import os
import sys
import subprocess
import json
from flask import Flask, request, render_template, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = "pea-seed-secret"   # needed for flash messages

# Folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# ROUTE: Index Page
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# ROUTE: Upload & Process File
# -----------------------------
@app.route('/pea_seed_analyzer', methods=['GET','POST'])
def upload_and_process_file():
    if 'imagefile' not in request.files:
        flash("No file uploaded.")
        return redirect(url_for('index'))

    file = request.files['imagefile']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    if file:
        # Generate unique name
        ext = os.path.splitext(file.filename)[1]
        unique_id = uuid.uuid4().hex
        input_filename = f"{unique_id}input{ext}"
        base_filename = os.path.splitext(input_filename)[0]
        input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_filepath)

        # Run ML script
        output_directory = OUTPUT_FOLDER
        try:
            subprocess.run(
                [sys.executable, "PRED_apk_d_2.py", input_filepath, output_directory, base_filename],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            flash("Error during processing.")
            print("STDOUT:", e.stdout.decode())
            print("STDERR:", e.stderr.decode())
            return redirect(url_for('index'))

        # Path to results.json inside the output subdir
        results_dir = os.path.join(output_directory, base_filename)
        results_json_path = os.path.join(results_dir, "results.json")

        if not os.path.exists(results_json_path):
            flash("Results not found.")
            return redirect(url_for('index'))

        # Load results.json
        with open(results_json_path, "r") as jf:
            results = json.load(jf)

        # Convert paths to URL-friendly format
        def path_to_url(path):
            return path.replace("\\", "/")
        
        # Send data to results.html
        return render_template("result.html",
                               input_image_url=path_to_url(results['uploaded_image']),
                               image_output_url=path_to_url(results['classified_image']),
                               csv_url=path_to_url(results['csv']),
                               pdf_url=path_to_url(results['pdf']),
                               total_seeds=results['total_seeds'],
                               healthy=results['healthy'],
                               infected=results['infected'],
                               confidence = results['confidence'])

# Route Protection
@app.route('/results')
def results():
    if 'results' not in session:
        # User tried to access /results directly
        return redirect(url_for('index'))

    results_data = session.pop('results')  # remove after fetching
    return render_template('result.html', **results_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
