from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import os
import uuid
from Modes import Modes
from ImageAugmentor import ImageAugmentor

class Utilities:
    @staticmethod
    def open_image(image_path):
        try:
            Utilities.open_file(image_path)
        except FileNotFoundError as e:
            Utilities.show_error_message(f"Ошибка: {str(e)}")
            return None
        except Exception as e:
            Utilities.show_error_message(f"Неизвестная ошибка при проверке пути: {str(e)}")
            return None

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Файл не является изображением или поврежден")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            Utilities.show_error_message(f"Ошибка при загрузке изображения: {str(e)}")
            return None

    @staticmethod
    def open_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")   

    @staticmethod
    def show_error_message(message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Ошибка")
        msg_box.setText(message)
        msg_box.exec_()     

    @staticmethod
    def show_message(message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("сообщение")
        msg_box.setText(message)
        msg_box.exec_() 

    @staticmethod
    def determine_mode(directory):
        images_dir = os.path.join(directory, "images")
        labels_dir = os.path.join(directory, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            return Modes.IMAGES_WITH_LABELS
        else:
            return Modes.ONLY_IMAGES

    @staticmethod
    def process_labels(mode, directory, image_path):
        if mode != Modes.IMAGES_WITH_LABELS:
            return None, None

        label_path = Utilities.get_labels_path(directory, image_path)
        if not os.path.exists(label_path):
            return None, None

        try:
            bboxes, labels = Utilities.read_yolo_labels(label_path)
            return bboxes, labels
        except Exception as e:
            Utilities.show_error_message(f"Error reading labels from {label_path}: {e}")
            return None, None

    @staticmethod
    def get_labels_path(directory, image_path):
        return os.path.join(
            directory, "labels", 
            os.path.basename(image_path).replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt").replace(".bmp", ".txt")
        )
    
    @staticmethod
    def read_yolo_labels(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()

        bboxes = []
        labels = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append(int(class_id))
            bboxes.append([x_center, y_center, width, height])

        return bboxes, labels
    
    @staticmethod
    def save_yolo_labels(label_path, bboxes, labels):
        with open(label_path, 'w') as file:
            for bbox, label in zip(bboxes, labels):
                x_center, y_center, width, height = bbox
                file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    @staticmethod
    def draw_boxes(image, bboxes, color=(0, 255, 0), thickness=2):
        image_with_bboxes = image.copy()
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            h, w, _ = image_with_bboxes.shape
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)

            cv2.rectangle(image_with_bboxes, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        return image_with_bboxes
    
    @staticmethod
    def numpy_to_pixmap(image, preview_width, preview_height):
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        q_image = QPixmap(QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888))
        return q_image.scaled(preview_width, preview_height, Qt.KeepAspectRatio)
    
    @staticmethod
    def attempt_augmentation(pipeline, image, bboxes, labels, attempts=3):
        for attempt in range(attempts):
            try:
                return True, *ImageAugmentor.augment_image(image, pipeline, bboxes, labels)
            except Exception as e:
                if attempt < attempts - 1:
                    continue
                return False, image, None, None
            
    @staticmethod
    def save_augm(mode, iter, image_path, output_images_dir, output_labels_dir, augmented_image, augmented_bboxes, augmented_labels):
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        new_image_name = f"aug_{iter}_{os.path.basename(image_path)}"
        new_image_path = os.path.join(output_images_dir, new_image_name)

        Utilities.save_image(augmented_image, new_image_path)

        if mode == Modes.IMAGES_WITH_LABELS and augmented_bboxes and augmented_labels:
            new_label_name = f"aug_{iter}_{base_name}.txt"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            Utilities.save_yolo_labels(new_label_path, augmented_bboxes, augmented_labels)
            
    @staticmethod        
    def save_image(image, path):
        try:
            if not os.path.splitext(path)[1]:
                file_name = f"{uuid.uuid4()}.png"
                path = os.path.join(path, file_name)
                
            cv2.imwrite(path, image)
            return True
        except Exception as e:
            Utilities.show_error_message(f"Ошибка сохранения: {e}")
            return False        