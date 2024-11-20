import albumentations as A
from Modes import Modes

class ImageAugmentor:
    @staticmethod
    def augment_image(image, pipeline, bboxes=None, labels=None):
        data = {"image": image}
        if bboxes is not None and labels is not None:
            data["bboxes"] = bboxes
            data["labels"] = labels

        augmented = pipeline(**data)
        return augmented['image'], augmented.get('bboxes', None), augmented.get('labels', None)
    
    @staticmethod
    def update_pipeline(settings, mode):
        transforms = []
        for aug, config in settings.items():
            if config["enabled"]:
                prob = config["probability"]
                transforms.append(getattr(A, aug)(p=prob))

        if mode == Modes.ONLY_IMAGES:
            return A.Compose(transforms)
        
        elif mode == Modes.IMAGES_WITH_LABELS:
            return A.Compose(
                transforms, 
                bbox_params=A.BboxParams(
                    format='yolo', 
                    label_fields=['labels']
                    ),
                )
