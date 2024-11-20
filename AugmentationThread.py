from PyQt5.QtCore import QThread, pyqtSignal
import numpy
import time
from Utilities import Utilities
from concurrent.futures import ThreadPoolExecutor, as_completed

class AugmentationThread(QThread):
    progress = pyqtSignal(int)          # Сигнал для обновления прогресс-бара
    error = pyqtSignal(str)             # Сигнал для отправки ошибок
    finished = pyqtSignal(int, int, float)     # Сигнал завершения
    progress_preview = pyqtSignal(numpy.ndarray, object, numpy.ndarray, object)     # Сигнал для превью
    ENABLE_PREVIEW = True

    def __init__(self, directory, image_paths, labels_dir, output_images_dir, output_labels_dir, pipeline, mode, augmentations_per_image, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.output_images_dir = output_images_dir
        self.output_labels_dir = output_labels_dir
        self.pipeline = pipeline
        self.mode = mode
        self.augmentations_per_image = augmentations_per_image
        self._is_running = True

    def process_image(self, image_path):
        if not self._is_running:
            return False

        try:
            image = Utilities.open_image(image_path)
            bboxes, labels = Utilities.process_labels(self.mode, self.directory, image_path)

            for i in range(self.augmentations_per_image):
                ok, augmented_image, augmented_bboxes, augmented_labels = Utilities.attempt_augmentation(
                    self.pipeline, 
                    image, 
                    bboxes, 
                    labels,
                )

                if ok:
                    Utilities.save_augm(
                        self.mode, 
                        i, 
                        image_path, 
                        self.output_images_dir, 
                        self.output_labels_dir, 
                        augmented_image, 
                        augmented_bboxes, 
                        augmented_labels,
                    )

                    if self.ENABLE_PREVIEW:
                        self.progress_preview.emit(image, bboxes, augmented_image, augmented_bboxes)

            return True
        except Exception as e:
            self.error.emit(f"Error processing {image_path}: {e}")
            return False

    def run(self):
        start_time = time.time()

        total_images = len(self.image_paths)
        total_iterations = total_images * self.augmentations_per_image
        iteration = 0

        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_image = {executor.submit(self.process_image, image_path): image_path for image_path in self.image_paths}

            for future in as_completed(future_to_image):
                try:
                    success = future.result()
                    if success:
                        iteration += self.augmentations_per_image
                        self.progress.emit(int((iteration / total_iterations) * 100))
                except Exception as e:
                    image_path = future_to_image[future]
                    self.error.emit(f"Error processing {image_path}: {e}")

        end_time = time.time()  
        time_elapsed = end_time - start_time

        self.finished.emit(iteration, total_iterations, time_elapsed)

    def stop(self):
        self._is_running = False