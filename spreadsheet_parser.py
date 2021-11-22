from pdf2image import convert_from_path
from pytesseract import pytesseract
import cv2
import numpy as np
import re
import csv
import sys
import math

def __point_ordering__(data):
    data_array = []
    threshold = 15
    while len(data) > 0:
        lefttoppoint = sorted(data, key=lambda d: d[0][0][0] + d[0][0][1])[0][0][0]
        righttoppoint = sorted(data, key=lambda d: d[0][0][0] - d[0][0][1])[-1][0][0]
        tlp = np.array(lefttoppoint)
        trp = np.array(righttoppoint)
        row_data = []
        remaining_data = []
        for k in data:
            p = np.array(k[0][0])
            d = np.linalg.norm(np.cross(trp-tlp, tlp-p))/np.linalg.norm(trp-tlp)
            if d < threshold or math.isnan(d):
                row_data.append(k)
            else:
                remaining_data.append(k)
        row_data = sorted(row_data, key=lambda c: c[0][0][0], reverse=False)
        data_array.append(row_data)
        data = remaining_data
    return data_array

def __parse_image__(file, lang):
    i = 1
    page_list = []
    page_data = []
    if file.endswith(".pdf"):
        pdf_file_path = file
        pages = convert_from_path(r"{}".format(pdf_file_path), 300, size=(1200,None))
        for page in pages:
            image_name = "temp_" + str(i) + ".jpg"
            page.save(image_name, "JPEG")
            page_list.insert(0, image_name)
            i = i + 1
    else:
        page_list.append(file)
    for image_path in page_list:
        dataset = []
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 6)
        cnts, hierachy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        masking_contour = np.zeros(im.shape[:2],dtype='uint8')
        for cnt in cnts:
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(approx)
                if w > 20 and h > 20:
                    grid = cv2.line(masking_contour, (0, y), (im.shape[1], y), color=255, thickness=1)
                    grid = cv2.line(masking_contour, (x, 0), (x, im.shape[0]), color=255, thickness=1)
                    grid = cv2.line(masking_contour, (0, y+h), (im.shape[1], y+h), color=255, thickness=1)
                    grid = cv2.line(masking_contour, (x+w, 0), (x+w, im.shape[0]), color=255, thickness=1) 
        cnts, hierachy = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            offset = -1
            if w > 20 and h > 20 and w < im.shape[1] * 0.95 and h < im.shape[0] * 0.95:
                image = cv2.rectangle(im, (x-offset,y-offset), (x+w+offset, y+h+offset), color=(255,0,255), thickness=2)
                cords = [(x-offset,y-offset), (x+w+offset, y+h+offset)]
                cropped_im = gray[cords[0][1]:cords[1][1], cords[0][0]:cords[1][0]]
                text = str(pytesseract.image_to_string(cropped_im, config=f'-l {lang} --psm 6'))
                dataset.insert(0, [cords, text])
        page_data.insert(0, __point_ordering__(dataset))
    return page_data



def __clean__(text):
    text = re.sub(r'[_\n\x0c]', '', text)
    return text

def main(argv):
    file = argv[0]
    output = argv[1]
    if len(argv) > 2:
        lang = argv[2]
    else:
        lang = 'eng'
    print("file", file)
    print("output", output)
    data = __parse_image__(file, lang)
    sorted_array = []
    for page in data:
        for row in page:
            sorted_array.append([__clean__(col[1]) for col in row])
    with open(output, "w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(sorted_array)
        

if __name__ == "__main__":
    main(sys.argv[1:])