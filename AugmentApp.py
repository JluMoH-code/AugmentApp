import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QWidget, QProgressBar
)
from PyQt5.QtCore import Qt
from AugmentationSettingsDialog import AugmentationSettingsDialog
from Modes import Modes
from ImageAugmentor import ImageAugmentor
from AugmentationThread import AugmentationThread
from Utilities import Utilities

class DataAugmentationApp(QMainWindow):
    PREVIEW_HEIGHT, PREVIEW_WIDTH = 400, 400
    WINDOW_TITLE = "Приложение аугментации"
    DEFAULT_PROBABILITY = 0.3
    DEFAULT_AUG_PER_IMAGE = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE) 
        self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)

        self.directory = None
        self.image_paths = []
        self.current_index = 0
        self.pipeline = None
        self.mode = Modes.ONLY_IMAGES
        self.augmentations_per_image = self.DEFAULT_AUG_PER_IMAGE

        # Состояния параметров аугментации (все включены по умолчанию)
        self.augmentation_settings = {
            aug: {"enabled": True, "probability": self.DEFAULT_PROBABILITY} for aug in [
                "Affine", "CLAHE", "ChannelShuffle", "ChromaticAberration",
                "CoarseDropout", "ColorJitter", "D4", "Downscale",
                "HueSaturationValue", "ISONoise", "Morphological",
                "MotionBlur", "OpticalDistortion",
                "PixelDropout", "RGBShift", "RandomBrightnessContrast",
                "RandomGamma", "RandomGravel", "RandomRain", "Sharpen", "Spatter"
            ]
        }
        self.pipeline = ImageAugmentor.update_pipeline(self.augmentation_settings, self.mode)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.dir_label = QLabel("Активная директория: None")
        self.select_dir_button = QPushButton("Выбрать директорию")
        self.select_dir_button.clicked.connect(self.select_directory)
        layout.addWidget(self.dir_label)
        layout.addWidget(self.select_dir_button)

        self.settings_button = QPushButton("Настройка аугментации")
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)

        # Добавляем кнопку "Старт аугментации"
        self.start_button = QPushButton("Старт аугментации")
        self.start_button.clicked.connect(self.start_augmentation)
        layout.addWidget(self.start_button)

        # Кнопка "Остановить аугментацию" (по умолчанию скрыта)
        self.stop_button = QPushButton("Остановить аугментацию")
        self.stop_button.clicked.connect(self.stop_augmentation)
        self.stop_button.setVisible(False)
        layout.addWidget(self.stop_button)

        # Добавляем прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)  # Начальное значение
        layout.addWidget(self.progress_bar)

        self.image_layout = QHBoxLayout()
        self.original_image_label = QLabel("Исходное изображение")
        self.augmented_image_label = QLabel("Аугментированное изображение")
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.augmented_image_label)
        layout.addLayout(self.image_layout)

        self.nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Предыдущее")
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button = QPushButton("Следующее")
        self.next_button.clicked.connect(self.next_image)
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.next_button)
        layout.addLayout(self.nav_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_augmentation(self):
        if not self.directory:
            Utilities.show_error_message("Выберите директорию перед началом аугментации.")
            return

        images_dir = os.path.join(self.directory, "images")
        labels_dir = os.path.join(self.directory, "labels")
        output_images_dir = os.path.join(self.directory, "augmented_images")
        output_labels_dir = os.path.join(self.directory, "augmented_labels")

        # Создаем директории для сохранения результатов
        os.makedirs(output_images_dir, exist_ok=True)
        if self.mode == Modes.IMAGES_WITH_LABELS:
            os.makedirs(output_labels_dir, exist_ok=True)

        # Инициализируем поток
        self.augmentation_thread = AugmentationThread(
            self.directory,
            self.image_paths,
            labels_dir,
            output_images_dir,
            output_labels_dir,
            self.pipeline,
            self.mode,
            self.augmentations_per_image,
        )
        self.augmentation_thread.progress.connect(self.progress_bar.setValue)
        self.augmentation_thread.error.connect(Utilities.show_error_message)
        self.augmentation_thread.finished.connect(self.on_augmentation_finished)
        self.augmentation_thread.progress_preview.connect(self.preview_progress)

        # Блокируем интерфейс
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stop_button.setVisible(True)

        # Запускаем поток
        self.augmentation_thread.start()

    def on_augmentation_finished(self, count, total_count, time):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stop_button.setVisible(False)
        Utilities.show_message(f"Аугментация завершена!\nПолучено изображений: {count}/{total_count}\nВремени затрачено: {time:.2f} с")

    def stop_augmentation(self):
        if self.augmentation_thread and self.augmentation_thread.isRunning():
            self.augmentation_thread.stop()
            self.augmentation_thread.wait()
            self.on_augmentation_stopped()

    def on_augmentation_stopped(self):
        self.start_button.setEnabled(True)
        self.stop_button.setVisible(False)
        Utilities.show_message("Аугментация была остановлена пользователем.")

    def preview_progress(self, orig_image, orig_bboxes, aug_image, aug_bboxes):
        self.display_image(orig_image, orig_bboxes, self.original_image_label)
        self.display_image(aug_image, aug_bboxes, self.augmented_image_label)

    def select_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Выберите директорию")
        if self.directory:
            self.dir_label.setText(f"Активная директория: {self.directory}")
            
            self.mode = Utilities.determine_mode(self.directory)
            if self.mode == Modes.IMAGES_WITH_LABELS:
                self.image_paths = [
                    os.path.join(self.directory, "images", f) for f in os.listdir(os.path.join(self.directory, "images"))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ]
            elif self.mode == Modes.ONLY_IMAGES:
                self.image_paths = [
                    os.path.join(self.directory, f) for f in os.listdir(self.directory)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ]
            
            self.current_index = 0
            self.pipeline = ImageAugmentor.update_pipeline(self.augmentation_settings, self.mode)
            self.show_image_pair()

    def open_settings(self):
        dialog = AugmentationSettingsDialog(self, self.augmentation_settings, self.augmentations_per_image)
        if dialog.exec_():
            self.augmentation_settings, self.augmentations_per_image = dialog.get_updated_settings()
            self.pipeline = ImageAugmentor.update_pipeline(self.augmentation_settings, self.mode)
            self.show_image_pair()

    def show_image_pair(self):
        if not self.has_valid_image_paths():
            return

        original_path = self.image_paths[self.current_index]
        
        image = Utilities.open_image(original_path)
        if image is None:
            return

        bboxes, labels = Utilities.process_labels(self.mode, self.directory, original_path)
        
        self.display_image(image, bboxes, self.original_image_label)

        ok, augmented_image, augmented_bboxes, augmented_labels = Utilities.attempt_augmentation(self.pipeline, image, bboxes, labels)
        
        self.display_image(augmented_image, augmented_bboxes, self.augmented_image_label)

        self.adjust_widget_sizes()

    def has_valid_image_paths(self):
        if not self.image_paths: return False
        else: return self.image_paths    

    def display_image(self, image, bboxes, label_widget):
        if bboxes:
            image = Utilities.draw_boxes(image, bboxes)

        pixmap = Utilities.numpy_to_pixmap(image, self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT).scaled(
            self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT, Qt.KeepAspectRatio
        )
        label_widget.setPixmap(pixmap)

    def adjust_widget_sizes(self):
        self.original_image_label.adjustSize()
        self.augmented_image_label.adjustSize()
        self.adjustSize()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image_pair()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image_pair()

if __name__ == "__main__":
    app = QApplication([])
    window = DataAugmentationApp()
    window.show()
    app.exec_()
