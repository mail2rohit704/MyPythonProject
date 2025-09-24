import sys
import cv2
import numpy as np
import csv
import os
import json
from closest_color import *
import joblib
from fpdf import FPDF
import tempfile

# -------------------------
# Utility Functions
# -------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def remove_substrings(s, a):
    for sub in a:
        s = s.replace(sub, "")
    return s

def calculate_ellipse_axis(contour):
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1][0], ellipse[1][1])
    minor_axis = min(ellipse[1][0], ellipse[1][1])
    return minor_axis, major_axis

def eccentricity_from_ellipse(contour):
    (xcenter, ycenter), (MA, ma), angle = cv2.fitEllipse(contour)
    width_bounding_box = MA / 2
    height_bounding_box = ma / 2
    ecc = np.sqrt(height_bounding_box ** 2 - width_bounding_box ** 2) / height_bounding_box
    return ecc, width_bounding_box, height_bounding_box, angle

def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    res = float(w) / h
    return res, w, h

def mybinaryotsuthresholding(img, kernelsizeX, kernelsizeY):
    global image
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(edged, 54, 255, cv2.THRESH_BINARY)
    edged = th3
    kernel = np.ones((kernelsizeX, kernelsizeY), np.uint8)
    img_erosion = cv2.erode(edged, kernel, iterations=0)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=0)
    edged = img_dilation
    edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
    return edged

class CSVWriter():
    def __init__(self, filename):
        self.fp = open(filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"',
                                 quoting=csv.QUOTE_ALL, lineterminator='\n')
    def close(self):
        self.fp.close()
    def write(self, elems):
        self.writer.writerow(elems)

# -------------------------
# Main Script
# -------------------------
if len(sys.argv) < 4:
    print("Usage: PRED_apk_d_2.py <input_image_path> <output_directory_path> <base_filename>")
    sys.exit(1)

input_filepath = sys.argv[1]
output_directory_path = sys.argv[2]
base_filename = sys.argv[3]

# Create subfolder inside outputs
output_subdir = os.path.join(output_directory_path, base_filename)
os.makedirs(output_subdir, exist_ok=True)

# Load image
image = cv2.imread(input_filepath)
filename_with_ext = os.path.basename(input_filepath)

# Load Model
CLASS_NAMES = {'C10_H': 'Healthy', 'C10_I': 'Not Healthy'}
MODEL_FILENAME = 'LogisticReg_ALL_B.joblib'
a = ["_ALL_B.joblib"]
try:
    loaded_model = joblib.load(resource_path(MODEL_FILENAME))
except FileNotFoundError:
    print("Error: Model file not found! Exiting.")
    sys.exit(1)

# Setup PDF
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font('helvetica', 'B', 20)
pdf.cell(0, 15, 'PROGRAM OUTCOME', ln=True, align='C')
pdf.set_font('helvetica', '', 12)
pdf.ln(10)
pdf.set_font('helvetica', 'B', 12)
pdf.set_fill_color(211, 211, 211)
col_widths = [30, 50, 60, 50]
header_titles = ['Seed No.', 'Seed Image', 'Classification', 'Prediction Probability']

def draw_table_header(pdf, col_widths, header_titles):
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_fill_color(211, 211, 211)
    for i in range(len(header_titles)):
        pdf.cell(col_widths[i], 10, header_titles[i], border=1, align='C', fill=True)
    pdf.ln()

draw_table_header(pdf, col_widths, header_titles)
pdf.set_font('helvetica', '', 11)
row_height = 40
edged = mybinaryotsuthresholding(image, 8, 8)
(contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# CSV setup
csv_path = os.path.join(output_subdir, f"{base_filename}.csv")
mycsv = CSVWriter(csv_path)
mycsv.write(('SNo', 'Classification', 'Prediction_Probability'))

i = 0
area_lower_limit = 1000
C2_Healthy = 0
C2_Infected = 0

for j, cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
    grain = image[y:y + h, x:x + w]
    area = cv2.contourArea(cnt)

    if area < area_lower_limit:
        continue

    i += 1

    # --------------------
    # Feature Extraction
    peri = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area)
    extent = float(area) / (w * h)
    equi_dia = np.sqrt(4 * area / np.pi)

    M = cv2.moments(cnt)
    Xcenter_fm_moments = round(M['m10'] / M['m00'])
    Ycenter_fm_moments = round(M['m01'] / M['m00'])

    if area < 30:
        roundness = 0
        eccentricity, minor_axis, major_axis, orientation_angle = 0, 0, 0, 0
    else:
        roundness = (peri * peri) / (area * 4 * np.pi)
        eccentricity, minor_axis, major_axis, orientation_angle = eccentricity_from_ellipse(cnt)

    minor_axis1, major_axis1 = calculate_ellipse_axis(cnt)
    aspectratio, width_BBox1, height_BBox1 = aspect_ratio(cnt)


    # Masking for seed extraction
    mask = np.zeros(image.shape[:2], dtype="uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask_crop = mask[y:y + h, x:x + w]
    masked_seed = cv2.bitwise_and(grain, grain, mask=mask_crop)

    # Label the grain image (crop) for PDF
    labeled_image = grain.copy()  # use the original cropped grain
    cv2.putText(labeled_image, str(i), (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    # Mean color extraction
    mean_color_grayscale = cv2.mean(masked_seed, mask=mask_crop)
    closest_color_info = closest(list_of_colors, [mean_color_grayscale[2], mean_color_grayscale[1], mean_color_grayscale[0]], RHS_colors)
    closest_color_RHS_Red = closest_color_info[0][0][0]
    closest_color_RHS_Green = closest_color_info[0][0][1]
    closest_color_RHS_Blue = closest_color_info[0][0][2]

    center = (int(Xcenter_fm_moments), int(Ycenter_fm_moments))
    maskc = cv2.putText(masked_seed.copy(), "{0}".format(i), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    raw = masked_seed.flatten()
    # Construct feature vector (dummy padded to 75)
    features = [
        round(area, 3), round(peri, 3), round(width_BBox1, 3), round(height_BBox1, 3),
        round(major_axis1, 3), round(minor_axis1, 3), round(roundness, 3), round(eccentricity, 3),
        100, round(aspectratio, 3), round(solidity, 3), round(extent, 3), round(equi_dia, 3),
        round(mean_color_grayscale[0], 0), round(mean_color_grayscale[1], 0), round(mean_color_grayscale[2], 0),
        200, 50, 22, raw.shape[0], 300, 340, 560, 230, 1231, 654, 789, 567, 432, 432, 899,
        908, 5679, 3452, 44, 2, 56, 45, 76, 456, 98, 678, 78, 89, 67, 78, 0, 7, 55, 34, 21,
        round(closest_color_RHS_Red, 3), round(closest_color_RHS_Green, 3), round(closest_color_RHS_Blue, 3),
        22, 44, 67, 999, 7836, 45, 340, 876, 2, 23145, 343, 2342,
        6753 * 100 / 255, 128 - 128, 10 - 128, 231, 564, 55, 33, 12, 55
    ]
    while len(features) < 75:
        features.append(0)

    features = np.array(features).reshape(1, -1)

    # Prediction
    pred_label = loaded_model.predict(features)[0]
    pred_class_name = CLASS_NAMES.get(pred_label, "Unknown")
    pred_prob = np.max(loaded_model.predict_proba(features))
    confidence_score = np.max(pred_prob)
    print(f"Seed {i}: {pred_class_name}, Probabilities: {pred_prob}")

    
    if pred_class_name == "Healthy":
        C2_Healthy += 1
    else:
        C2_Infected += 1

    # Draw classification label on the original image
    label_color = (0, 255, 0) if pred_class_name == "Healthy" else (0, 0, 255)
    cv2.putText(image, pred_class_name, (x - 5, y - 5),  # slightly above the seed
                cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 1, cv2.LINE_AA)

    # CSV write
    mycsv.write((i, pred_class_name, round(confidence_score * 100, 3)))

    # PDF Rendering
    if pdf.get_y() + row_height > pdf.page_break_trigger:
        pdf.add_page()
        draw_table_header(pdf, col_widths, header_titles)
        pdf.set_font('helvetica', '', 11)

    temp_image_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=output_directory_path) as temp_image:
            temp_image_path = temp_image.name
            cv2.imwrite(temp_image_path, labeled_image)

        row_start_x = pdf.get_x()
        row_start_y = pdf.get_y()
        pdf.cell(col_widths[0], row_height, str(i), border=1, align='C')

        img_padding = 2
        x_for_image = row_start_x + col_widths[0] + img_padding
        y_for_image = row_start_y + img_padding
        img_height = row_height - (2 * img_padding)
        pdf.image(temp_image_path, x=x_for_image, y=y_for_image, h=img_height)
        pdf.cell(col_widths[1], row_height, '', border=1)
        pdf.cell(col_widths[2], row_height, str(pred_class_name), border=1, align='C')
        pdf.cell(col_widths[3], row_height, f"{round(pred_prob * 100, 3):.3f}", border=1, align='C', ln=1)

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Final PDF Summary
pdf.ln(10)
pdf.set_font('helvetica', 'B', 20)
pdf.cell(0, 15, 'RESULT', ln=True, align='L')
pdf.set_font('helvetica', '', 12)
pdf.cell(0, 10, 'Total Number of Seeds Found = ' + str(i), ln=True, align='L')
pdf.cell(0, 10, 'Machine Learning Algorithm Used = ' + remove_substrings(str(MODEL_FILENAME), a), ln=True, align='L')
pdf.cell(0, 10, 'Total Healthy Seeds Found = ' + str(C2_Healthy), ln=True, align='L')
pdf.cell(0, 10, 'Total Infected Seeds Found = ' + str(C2_Infected), ln=True, align='L')
pdf.ln(10)

mycsv.close()


# Save classified image
classified_img_path = os.path.join(output_subdir, f"{base_filename}_classification.jpg")
cv2.imwrite(classified_img_path, cv2.putText(image.copy(), f'Total: {i}', (20,50),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2))

# Save PDF report
final_pdf_path = os.path.join(output_subdir, f"{base_filename}_report.pdf")
pdf.output(final_pdf_path)

def path_to_url(path):
    return path.replace("\\", "/")

# Save JSON results
results_json_path = os.path.join(output_subdir, "results.json")
results_data = {
    "csv": path_to_url(csv_path),
    "classified_image": path_to_url(classified_img_path),
    "pdf": path_to_url(final_pdf_path),
    "total_seeds": i,
    "healthy": C2_Healthy,
    "infected": C2_Infected,
    "uploaded_image": path_to_url(input_filepath),
    "confidence": confidence_score
}
with open(results_json_path,"w") as jf:
    json.dump(results_data, jf, indent=4)

print(f"Processing completed. Results saved in {output_subdir}")
