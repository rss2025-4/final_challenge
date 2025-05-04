import random

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def _label_to_color(label):
    random.seed(label)  # Use label as seed to generate a stable color
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


class Detector:
    def __init__(self, yolo_dir="/root/yolo", from_tensor_rt=True, threshold=0.5):
        # local import
        # from ultralytics import YOLO
        from ultralytics import YOLOE

        # cls = YOLO
        cls = YOLOE
        
        self.threshold = threshold
        self.yolo_dir = yolo_dir
        if from_tensor_rt:
            self.model = cls(f"{self.yolo_dir}/yolo11n.engine", task="detect")
        else:
            # self.model = cls(f"{self.yolo_dir}/yolo11n.pt", task="detect")
            # self.model = cls(f"{self.yolo_dir}/yoloe-11s-seg-pf.pt", task="detect")
            # self.model = cls(f"{self.yolo_dir}/yoloe-v8s-seg.pt") #, task="detect")
            self.model = cls(f"{self.yolo_dir}/yoloe-11s-seg.pt", task="detect")

            names = ["stop light", "amber light", "green light", "red light"]
            self.model.set_classes(names, self.model.get_text_pe(names))


            # model = YOLOE("yoloe-11l-seg.pt")

    
    def to(self, device):
        self.model.to(device)

    def predict(self, img, silent=True):
        """
        Note: img can be any of the following:
            Union[str, pathlib.Path, int, PIL.Image.Image, list, tuple, numpy.ndarray, torch.Tensor]

            Batch not supported.
            
        Runs detection on a single image and returns a list of
        ((xmin, ymin, xmax, ymax), class_label) for each detection
        above the given confidence threshold.
        """
        results = list(self.model(img, verbose=not silent))[0]
        boxes = results.boxes

        predictions = []
        # Iterate over the bounding boxes
        for xyxy, conf, cls_idx in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if conf.item() >= self.threshold:
                # Convert bounding box tensor to Python floats
                x1, y1, x2, y2 = xyxy.tolist()
                # Map class index to class label using model/ results
                label = results.names[int(cls_idx.item())]
                predictions.append(((x1, y1, x2, y2), label))
        
        #convert original image to rgb
        original_image = results.orig_img
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB, original_image)
        
        return dict(predictions=predictions, original_image=original_image)
    
    def set_threshold(self, threshold):
        """
        Sets the confidence threshold for predictions.
        """
        self.threshold = threshold

    def draw_box(self, img, predictions, draw_all=False):
        """
        Draw bounding boxes on 'img'.

        :param img: The image to draw on (PIL.Image or NumPy array).
        :param predictions: A list of ((xmin, ymin, xmax, ymax), label).
        :param draw_all: If True, draw *all* bounding boxes.
                         If False, draw only the first one.
        :return: A PIL image with boxes and labels drawn.
        """
        if not predictions:
            return img  # No detections, return as is

        # Convert to PIL.Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)

        min_dim = min(img.width, img.height)
        scale_factor = (
            min_dim / 600.0
        )

        line_width = max(
            1, int(4 * scale_factor)
        )
        font_size = max(10, int(20 * scale_factor))
        text_offset = font_size + 3

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        print(f"Labels: {[x[-1] for x in predictions]}")

        if draw_all:
            for (x1, y1, x2, y2), label in predictions:
                color = _label_to_color(label)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                draw.text((x1, y1 - text_offset), label, fill=color, font=font)
        else:
            (x1, y1, x2, y2), label = predictions[0]
            color = _label_to_color(label)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            draw.text((x1, y1 - text_offset), label, fill=color, font=font)

        return img
    
    def id2name(self, i):
        """
        Converts a class index to a class name.
        """
        return self.model.names[i]
    
    @property
    def classes(self):
        return self.model.names
        
    
def demo():
    import os
    model = Detector(yolo_dir='/home/racecar/models', from_tensor_rt=False)
    model.set_threshold(0.5)

    
    
    img_path = f"{os.path.dirname(__file__)}/../../media/trafficlight_2.png" 
        
    img = Image.open(img_path)
    results = model.predict(img)
    
    predictions = results["predictions"]
    print(f"Predictions: {predictions}")
    original_image = results["original_image"]
        
    out = model.draw_box(original_image, predictions, draw_all=True)
    
    save_path = f"{os.path.dirname(__file__)}/demo_output6.png"
    out.save(save_path)
    print(f"Saved demo to {save_path}!")

    
    # # # crop the image to the bounding box
    # # pil_image = Image.open("your_image.jpg")

    # # Convert to NumPy array for slicing
    
    # image_np = np.array(img)
    # # Copy image and blacken it
    # output = np.zeros_like(image_np)

    # # Find the traffic light and paste only that part into the black canvas
    # for bbox, label in predictions:
    #     if label == 'traffic light':
    #         x1, y1, x2, y2 = map(int, bbox)
    #         output[y1:y2, x1:x2] = image_np[y1:y2, x1:x2]  # keep original pixels only in bbox region

    #         # Save the result
    #         result_img = Image.fromarray(output)
    #         save_path = os.path.join(os.path.dirname(__file__), "highlighted_traffic_light.png")
    #         result_img.save(save_path)
    #         print(f"Result saved to {save_path}")
    #         break
    
    # # orange_lower = np.array([0,225,140])  # hue saturation value
    # # orange_upper = np.array([30,300,300])
    # output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # image_hsv = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2HSV)

    # img = output_bgr

    # # lower_red1 = np.array([0, 100, 100])
    # # upper_red1 = np.array([10, 255, 255])
    # # bounding_box = ((0,0),(0,0))
    
    # # image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    
    # # cv2.imshow("image", image_hsv)
    # # cv2.waitKey(0)

    # # Apply dual red mask
    # # lower_red1 = np.array([0, 100, 100])
    # # upper_red1 = np.array([10, 255, 255])
    # # lower_red2 = np.array([160, 100, 100])
    # # upper_red2 = np.array([179, 255, 255])

    # lower_red1 = np.array([0, 180, 120])
    # upper_red1 = np.array([10, 255, 230])

    # lower_red2 = np.array([160, 180, 120])
    # upper_red2 = np.array([179, 255, 230])

    # # Split and equalize brightness (optional)
    # h, s, v = cv2.split(image_hsv)
    # clahe = cv2.createCLAHE(clipLimit=2.03, tileGridSize=(8, 8))
    # v_eq = clahe.apply(v)
    # hsv_eq = cv2.merge([h, s, v_eq])
    # mask1 = cv2.inRange(hsv_eq, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv_eq, lower_red2, upper_red2)
    # red_mask = cv2.bitwise_or(mask1, mask2)

    # cv2.imshow("Red Traffic Light Mask", red_mask)
    # cv2.waitKey(0)

    # # adjust brightness
    # h, s, v = cv2.split(image_hsv)
    # clahe = cv2.createCLAHE(clipLimit=2.03, tileGridSize=(8, 8))
    # v_eq = clahe.apply(v)
    # hsv_eq = cv2.merge([h, s, v_eq])
    # # image_print(hsv_eq)
    # # image_print(hsv_eq)
    # image_orange = cv2.inRange(hsv_eq, lower_red1, upper_red1)
    # # image_print(image_orange)
    # cv2.imshow("image_orange", image_orange)
    # cv2.waitKey(0)

        ######
    # kernel = np.ones((3,3), np.uint8)
    # kernel = np.ones((3,3), np.uint8)
    # for _ in range(20):
    #     image_orange = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    #     image_orange = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # # cv2.imshow("image_orange", image_orange)
    # # cv2.waitKey(0)

    # # distance transform
    # dist_transform = cv2.distanceTransform(image_orange, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # sure_fg = np.uint8(sure_fg)
    # # image_print(sure_fg)
    # # overlap
    # # print(image_orange)
    # unknown = cv2.subtract(image_orange, sure_fg)
    # # image_print(unknown)
    # # connected
    # _, markers = cv2.connectedComponents(sure_fg)
    # markers = markers + 1 
    # markers[unknown == 255] = 0
    # # image_print(sure_fg)
    # # watershed
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0] 

    # segmented_mask = np.zeros_like(image_orange)
    # segmented_mask[markers > 1] = 255
    # segmented_mask = np.zeros_like(image_orange)
    # segmented_mask[markers > 1] = 255

    # # image_print(segmented_mask)

    # contours, _ = cv2.findContours(segmented_mask,  
    # cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # largest_contour = max(contours, key=cv2.contourArea)
    # # print(contours[0].shape)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
    # # image_print(segmented_mask)
    # # image_print(img)
    # # image_print(img)
    # # print(len(contours))
    # x, y, w, h = cv2.boundingRect(largest_contour)
    # center_x = x + w / 2
    # center_y = y + h / 2

    # # Make rectangel slightly larger (optimized for test cases)
    # scale = 1.055
    # w_new = int(w * scale)
    # h_new = int(h * scale)
    # x_new = int(center_x - w_new / 2)
    # y_new = int(center_y - h_new / 2)
    # x_new = max(x_new, 0)
    # y_new = max(y_new, 0)
    # if x_new + w_new > img.shape[1]:
    #     w_new = img.shape[1] - x_new
    # if y_new + h_new > img.shape[0]:
    #     h_new = img.shape[0] - y_new

    # cv2.rectangle(img, (x_new, y_new), (x_new + w_new, y_new + h_new), 128, 2)
    # cv2.imshow("final", img)
    # cv2.waitKey(0)

	# image_print(img)
	# bounding_box = ((x_new, y_new), (x_new + w_new, y_new + h_new))



if __name__ == '__main__':    
    demo()
