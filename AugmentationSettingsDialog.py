from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, 
    QWidget, QDialog, QScrollArea, QFormLayout, QSlider
)
from PyQt5.QtCore import Qt

class AugmentationSettingsDialog(QDialog):
    WIDNOW_WIDTH, WINDOW_HEIGHT = 400, 600
    WINDOW_TITLE = "Настройка аугментации"

    def __init__(self, parent, current_settings, augmentations_per_image):
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(self.WIDNOW_WIDTH, self.WINDOW_HEIGHT)
        self.layout = QVBoxLayout(self)

        self.current_settings = current_settings  
        self.aug_widgets = {}
        self.advanced_visible = False 
        self.augmentations_per_image = augmentations_per_image

        self.initUI()

    def initUI(self):
        self.form_layout = QVBoxLayout()

        self.toggle_button = QPushButton("Показать доп. параметры")
        self.toggle_button.clicked.connect(self.toggle_advanced)
        self.layout.addWidget(self.toggle_button)

        self.augmentations_layout = QHBoxLayout()
        self.augmentations_label = QLabel(f"Аугментаций на изображение: {self.augmentations_per_image}")
        self.augmentations_slider = QSlider(Qt.Horizontal)
        self.augmentations_slider.setRange(1, 5)  # Задайте нужный диапазон
        self.augmentations_slider.setValue(self.augmentations_per_image)
        self.augmentations_layout.addWidget(self.augmentations_label)
        self.augmentations_layout.addWidget(self.augmentations_slider)
        self.augmentations_slider.valueChanged.connect(self.udpate_augmentation_label)
        self.layout.addLayout(self.augmentations_layout)

        # Мастер-настройка
        self.master_slider = QSlider(Qt.Horizontal)
        self.master_slider.setRange(1, 100)
        self.master_slider.setValue(30)  # Значение по умолчанию
        self.master_label = QLabel("Мастер-настройка: 0.30")
        self.master_slider.valueChanged.connect(self.update_all_probabilities)

        master_layout = QHBoxLayout()
        master_layout.addWidget(self.master_label)
        master_layout.addWidget(self.master_slider)

        self.layout.addLayout(master_layout)

        self.parameter_layout = QFormLayout()
        for aug, settings in self.current_settings.items():
            checkbox = QCheckBox(aug)
            checkbox.setChecked(settings["enabled"])

            # Ползунок для выбора вероятности
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)  
            slider.setValue(int(settings["probability"] * 100))  
            slider.setVisible(False) 

            # Метка для отображения текущего значения вероятности
            label = QLabel(f"{settings['probability']:.2f}")
            label.setVisible(False)

            slider.valueChanged.connect(
                lambda value, lbl=label: lbl.setText(f"{value / 100:.2f}")
            )

            self.aug_widgets[aug] = {
                "checkbox": checkbox,
                "slider": slider,
                "label": label,
            }

            row_layout = QHBoxLayout()

            left_container = QHBoxLayout()
            left_container.addWidget(checkbox)
            left_container.addStretch()  

            right_container = QHBoxLayout()
            right_container.addWidget(label)
            right_container.addWidget(slider)
            right_container.setAlignment(Qt.AlignRight)  

            row_layout.addLayout(left_container)
            row_layout.addLayout(right_container)

            self.parameter_layout.addRow(row_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(self.parameter_layout)
        scroll_area.setWidget(container)

        self.link_label = QLabel(
            '<a href="https://demo.albumentations.ai/">Примеры аугментаций</a>'
        )
        self.link_label.setOpenExternalLinks(True)  # Открывать ссылки в браузере
        self.link_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.link_label)

        self.layout.addWidget(scroll_area)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def update_all_probabilities(self, value):
        probability = value / 100.0
        self.master_label.setText(f"Мастер-настройка: {probability:.2f}")
        for widgets in self.aug_widgets.values():
            widgets["slider"].setValue(value)  
            widgets["label"].setText(f"{probability:.2f}")  

    def udpate_augmentation_label(self, value):
        self.augmentations_label.setText(f"Аугментаций на изображение: {value}")

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        for widgets in self.aug_widgets.values():
            widgets["slider"].setVisible(self.advanced_visible)
            widgets["label"].setVisible(self.advanced_visible)
        self.toggle_button.setText(
            "Скрыть доп. параметры" if self.advanced_visible else "Показать доп. параметры"
        )

    def get_updated_settings(self):
        updated_settings = {}
        for aug, widgets in self.aug_widgets.items():
            updated_settings[aug] = {
                "enabled": widgets["checkbox"].isChecked(),
                "probability": float(widgets["label"].text())
            }
        augmentations_per_image = self.augmentations_slider.value()
        return updated_settings, augmentations_per_image
