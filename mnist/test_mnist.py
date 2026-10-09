from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import csv
import time
import idx2numpy
import numpy as np

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import VehicleDifferentiableLogic


# Wrapper around the network to pass inputs of the correct dimension
class Classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image):
        return self.model(image.unsqueeze(0).unsqueeze(0)).squeeze(0)


loaded_model = torch.load("../models/pytorch/mnist-classifier.pth", weights_only=False)
loaded_model.eval()

model = Classifier(loaded_model)
networks = {"classifier": model}

spec_path = "mnist-robustness-single.vcl"

images_np = idx2numpy.convert_from_file("t2-images.idx")
labels_np = idx2numpy.convert_from_file("t2-labels.idx")

first_image = torch.tensor(images_np[0].astype(np.float32))
second_image = torch.tensor(images_np[1].astype(np.float32))
images = [first_image, second_image]

first_label = int(labels_np[0])
second_label = int(labels_np[1])
labels = [first_label, second_label]


def write_search_result(
    epsilon,
    perturbation, 
    perturbed_image, 
    output, 
    label_before_perturbation, 
    label_after_perturbation, 
    search_time
) -> None:
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/mnist_{timestamp}.csv"

    with open(results_filename, "w", newline="") as results_file:
        writer = csv.writer(results_file)

        # Write column names
        writer.writerow([
            "counterexample_found", 
            "epsilon", 
            "perturbation", 
            "perturbed_image", 
            "output", 
            "label_before_perturbation", 
            "label_after_perturbation", 
            "search_time"
        ])

        # Write result
        if perturbation is not None:
            writer.writerow([
                True,
                epsilon,
                perturbation.tolist(),
                perturbed_image.tolist(),
                output.tolist(),
                label_before_perturbation,
                label_after_perturbation,
                search_time
            ])
        else:
            writer.writerow([
                False,
                epsilon,
                None,
                None,
                None,
                label_after_perturbation,
                None,
                search_time
            ])


# Helper function to check that a generated perturbation is valid, correctly bounded by epsilon,
# and causes a misclassification
def robustness_violated(epsilon, perturbation, perturbed_image, output, training_label) -> bool:
    bounded_by_epsilon = bool((perturbation.abs() <= epsilon).all())
    valid_image = bool(((perturbed_image >= 0) & (perturbed_image <= 1)).all())

    advises_training_label = True
    training_label_score = output[training_label]
    for i in range(len(output)):
        if i != training_label and output[i] >= training_label_score:
            advises_training_label = False

    return ((bounded_by_epsilon and valid_image) and not advises_training_label)
    
			
def test_pytorch_search_robust_first_image() -> None:
    epsilon = 0.05 # Replace with desired value of epsilon

    counterexample_found = False
    search_results = None

	# Try to find an adversarial perturbation within 1 hour
    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            datasets={"trainingImages": images, "trainingLabels": labels},
            parameters={"epsilon": first_image.new_tensor(epsilon)},
            num_searches=1,
        )
        if len(search_results["robustSingleImage"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["robustSingleImage"][0]

    output_original = model(first_image)
    print(f"Model output before perturbation: {output_original}")
    label_original = output_original.argmax().item()
    print(f"Predicted label before perturbation: {label_original}")

    if counterexample_found:
        assert len(samples) == 1

        perturbation = samples[0]["perturbation"]
        perturbed_image = first_image - perturbation
  
        output_perturbation = model(perturbed_image)
        print(f"Model output after perturbation: {output_perturbation}")
        label_perturbation = output_perturbation.argmax().item()
        print(f"Predicted label after perturbation: {label_perturbation}")

        assert robustness_violated(epsilon, perturbation, perturbed_image, output_perturbation, first_label)

        write_search_result(epsilon, perturbation, perturbed_image, output_perturbation, label_original, label_perturbation, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result(epsilon, None, None, None, label_original, None, "TIMEOUT")
