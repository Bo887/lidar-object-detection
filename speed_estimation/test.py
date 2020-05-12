import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import speed_estimation
from PIL import Image

speed_list = {}
label_start = 0
images = []
no_label = []

for fname in sorted(glob.glob("*.png")):
    im = cv2.imread(fname)
    im = im[:, :, 0]

    boxes = []

    contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    speed_list, label_start = speed_estimation.parse_frame(boxes, speed_list, label_start, 10)

    t = np.zeros_like(im)
    for box in boxes:
        cv2.drawContours(t, [np.int0(box)], 0, (255), 2)

    img = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    no_label.append(im_pil)

    for objs in speed_list:
        if speed_list[objs].average_speed != 0:
            cv2.putText(
                t,
                str(round(speed_list[objs].average_speed, 1)*5),
                tuple(np.mean(speed_list[objs].current_position, axis=0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 255, 255),
                2,
                cv2.LINE_AA
            )

    img = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    images.append(im_pil)

    # show bounding boxes
    # plt.imshow(t)
    # show original image
    # plt.imshow(im)
    # plt.show()

images[0].save('outputs/tracking.gif', save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)
no_label[0].save('outputs/no_label.gif', save_all=True, append_images=no_label[1:], optimize=False, duration=400, loop=0)