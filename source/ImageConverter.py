import glob
import os
import cv2
import pandas as pd

path = "C:/Users/abepa/PycharmProjects/WordParse/images"  # path to images
xml_list = []
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']


def main():
    for image in glob.glob(path + '/*.jpg'):
        create_boxes(image)
    convert_to_csv()


def create_boxes(image):
    name = os.path.basename(image)
    name = os.path.splitext(name)
    img = cv2.imread(image)
    height_image, width_image, channels = img.shape
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    gray = cv2.medianBlur(gray, 5)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.erode(thresh, None, iterations=4)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 200:
            continue

        value = (str(name[0]), width_image, height_image, "machine_legible", x, y, x+w, y+h)
        xml_list.append(value)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Finally show the image
    # cv2.imshow("Output", img)
    # cv2.waitKey(0)


def convert_to_csv():
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv("test.csv", index=None)


if __name__ == '__main__':
    main()
