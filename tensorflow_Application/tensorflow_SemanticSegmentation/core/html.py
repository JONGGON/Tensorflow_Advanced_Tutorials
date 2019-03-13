import glob
import os

import cv2
import dominate
import numpy as np
from dominate.tags import *


class HTML(object):

    def __init__(self, model_name="Segmentation", path="masking_image", character_size=30):

        self.model_name = model_name
        self.character_size = character_size

        # 사이즈별로 정렬하는 경우
        # self.image_list = sorted(glob.glob("{}/*".format(path)), key=lambda path: int(os.path.getsize(path)))

        # 차 영상의 RMS 기준으로 정렬하는 경우 - 이게 jpg 이미지를 읽어오는 거라서 RMS 값이 변한다.
        # self.image_list = sorted(glob.glob("{}/*".format(path)), key=lambda path: self.key_func(path))

        # 이미지의 번호를 읽어오기
        self.image_list = sorted(glob.glob("{}/*".format(path)),
                                 key=lambda path: int(os.path.basename(path).split("_")[0]))

        self.doc = dominate.document(title="report")
        # 실행
        self.make_html()

    def key_func(self, path):

        image = cv2.imread(path)
        splited_image = np.split(image, 3, axis=1)  # 이미지를 가로축 기준 3개로 자름
        rms_error = np.sqrt(np.mean(np.square(splited_image[0])))
        return rms_error

    def make_html(self):
        self.t = table(border=1, bordercolor="#FFFFFF", bordercolorlight="#FFFFFF", bordercolordark="#FFFFFF",
                       style="border-spacing:0px;table-layout:fixed;background-color:white;", width=2048)
        self.doc.add(self.t)
        with self.t:
            with td():
                with caption():
                    h1(self.model_name,
                       style="text-align: center;background-color: #FFBB00; color: white;font-size: {}px; font-family:Comic Sans MS;".format(
                           self.character_size * 2))
            for i, image in enumerate(self.image_list):
                with th(style="background-color:#47C83E;"):
                    with b():
                        h2("file name",
                           style="color:white;text-align:center;font-size: {}px;font-family:Comic Sans MS;".format(
                               self.character_size))
                with th(style="background-color:#47C83E;"):
                    with b():
                        h2("Input",
                           style="color:yellow;text-align:center;font-size: {}px;font-family:Comic Sans MS;".format(
                               self.character_size))
                with th(style="background-color:#47C83E;"):
                    with b():
                        h2("target",
                           style="color:yellow;text-align:center;font-size: {}px;font-family:Comic Sans MS;".format(
                               self.character_size))
                with th(style="background-color:#47C83E;"):
                    with b():
                        h2("prediction",
                           style="color:yellow;text-align:center;font-size: {}px;font-family:Comic Sans MS;".format(
                               self.character_size))
                with tr():
                    with td(style="background-color:#47C83E;"):
                        name = os.path.basename(image)
                        p("{}".format(name),
                          style="color:white;text-align:center;font-size: {}px;font-family:Comic Sans MS;".format(self.character_size+20))
                    with td():
                        img(style="width:{}px;align:right;".format(1536),
                            src=os.path.abspath(image))

        html_file = 'Report.html'
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

if __name__ == '__main__':
    html = HTML()
else:
    print("HTML imported")
