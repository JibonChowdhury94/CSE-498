import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class HeatmapGenerator:
    def __init__(self, predictor, csv_file, model_class, size, dataset_path, cam_output_size, save_path, model_name, threshold=0.5):
        self.predictor = predictor
        self.csv_file = csv_file
        self.model_class = model_class
        self.model = self._get_model()
        self.model_name = model_name
#         self.target_layers = [self.model.model.features[-7]]
        self.target_layers = self._get_target_layers()
        self.size = size
        self.dataset_path = dataset_path
        self.cam_output_size = cam_output_size
        self.save_path = save_path
        self.threshold = threshold
        self.plot_name = "gradcam_heatmap.png"
        
    def _get_model(self):
        model = self.model_class(num_classes=len(self.predictor.label_columns))
        model_path = os.path.join(self.predictor.models_path, 'fold_1_model.pth') # Or any fold you'd like to visualize
        model.load_state_dict(torch.load(model_path))
        
        return model
    
    def _get_target_layers(self):
        target_layer_mapping = {
            'AlexNet': [self.model.model.features[-7]],
#             'MobileNetV2': [self.model.model.features[-7]],  # Last block of MobileNetV2
            'EfficientNetB0': [self.model.model.features[-2]],
        }

        # Return the target layers for the specific model, or a default layer
        return target_layer_mapping.get(self.model_name)  # Default layer
        
    
    def _get_prediction(self, input_tensor):
        output = self.model(input_tensor)
        return (output > self.threshold).float().squeeze()
        

    def _generate_heatmap(self, image_name: str) -> np.ndarray:
        input_image_path = os.path.join(self.dataset_path, image_name)
        input_image = Image.open(input_image_path).convert("RGB")
        rgb_img = np.array(input_image).astype(np.float32) / 255.0
        rgb_img_resized_for_cam = cv2.resize(rgb_img, (self.size, self.size))
        
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.unsqueeze(0)

        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img_resized_for_cam, grayscale_cam, use_rgb=True)
        return cv2.resize(visualization, (self.cam_output_size, self.cam_output_size))

    def _find_predictions(self):
        correct_predictions = []
        incorrect_predictions = []

        test_df = pd.read_csv(self.csv_file)

        for idx, (input_tensor, true_label) in enumerate(self.predictor.prepare_data_for_test()):
            predicted_label = self._get_prediction(input_tensor)

            actual_classes = [self.predictor.label_columns[i] for i, x in enumerate(true_label.squeeze().tolist()) if x == 1]
            predicted_classes = [self.predictor.label_columns[i] for i, x in enumerate(predicted_label.tolist()) if x == 1]

            if torch.all(predicted_label == true_label.squeeze()) and len(correct_predictions) < 3:
                correct_predictions.append((test_df['SOPInstanceUID_with_png'][idx], actual_classes, predicted_classes))

            elif not torch.all(predicted_label == true_label.squeeze()) and len(incorrect_predictions) < 3:
                incorrect_predictions.append((test_df['SOPInstanceUID_with_png'][idx], actual_classes, predicted_classes))

            if len(correct_predictions) == 3 and len(incorrect_predictions) == 3:
                break

        return correct_predictions, incorrect_predictions


    def plot_heatmaps(self):
        def format_classes(classes):
            formatted = []
            for cls in classes:
                formatted_cls = cls.replace("CalvarialFracture", "Calvarial Fracture").replace("OtherFracture", "Other Fracture")
                formatted.append(formatted_cls)
            return formatted

        def visualize_heatmap(ax, image_name, actual_classes, predicted_classes):
            actual_classes = format_classes(actual_classes or ["Normal"])
            predicted_classes = format_classes(predicted_classes or ["Normal"])
            heatmap = self._generate_heatmap(image_name)
            ax.imshow(heatmap)
            ax.set_title(f"Actual: {', '.join(actual_classes)} \n Predicted: {', '.join(predicted_classes)}", fontsize=12)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Removing tick marks
            ax.axis('off')

        correct_predictions, incorrect_predictions = self._find_predictions()

        # Setting up style for better visuals
        plt.style.use('seaborn-darkgrid')

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Prediction Heatmaps', fontsize=16, y=1.02)  # Adjusted y for better title spacing

        for i, data in enumerate(correct_predictions):
            visualize_heatmap(axs[0, i], *data)

        for i, data in enumerate(incorrect_predictions):
            visualize_heatmap(axs[1, i], *data)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjusted rect for better spacing
        plt.savefig(os.path.join(self.save_path, self.plot_name), dpi=300)
        plt.show()
