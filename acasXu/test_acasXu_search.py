from datetime import datetime
from math import pi

import torch
import torch.nn as nn
import csv
import time

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import VehicleDifferentiableLogic

# Inputs
distanceToIntruder = 0
angleToIntruder = 1
intruderHeading = 2
speed = 3
intruderSpeed = 4

# Outputs
clearOfConflict = 0
weakLeft = 1
weakRight = 2
strongLeft = 3
strongRight = 4

minimum_input_values = [0.0, -pi, -pi, 100.0, 0.0]
maximum_input_values = [60261.0, pi, pi, 1200.0, 1200.0]
mean_scaling_values = [19791.091, 0.0, 0.0, 650.0, 600.0]

# Network wrapper to match the specification's input and output dimensions
class ACASXu(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.unsqueeze(0)).squeeze(0)

loaded_model = torch.load("../models/pytorch/acasXu_1_7.pth", weights_only=False)
loaded_model.eval()

model = ACASXu(loaded_model)
model_name = "acasXu_1_7"

spec_path = "acasXu.vcl"

########################################################################

# Function to write the result for each property to a CSV file

########################################################################

def write_search_result(property, search_result, model, search_time):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_filename = f"results/search/acasXu_search_{property}_{timestamp}.csv"

    with open(results_filename, "w", newline="") as results_file:
        writer = csv.writer(results_file)

        # Write column names
        writer.writerow(["counterexample_found", "input", "output", "search_time"])

        # Write result
        if len(search_result) > 0:
            input = search_result[0]["x"]
            output = model(normalise(input))
            writer.writerow([
                True,
                input.tolist(),
                output.tolist(),
                search_time
            ])
        else:
            writer.writerow([
                False,
                None,
                None,
                search_time
            ])

########################################################################

# Helper functions to check that a generated counter-example for a 
# property actually violates it

########################################################################

def valid_input(input) -> bool:
    for i in range(len(input)):
        if minimum_input_values[i] <= input[i] <= maximum_input_values[i]:
            continue
        else:
            return False
    return True


def normalise(input) -> torch.Tensor:
    normalised_input = []
    for i in range(len(input)):
        normalised = (input[i] - mean_scaling_values[i]) / (maximum_input_values[i] - minimum_input_values[i])
        normalised_input.append(normalised)

    return torch.Tensor(normalised_input)


def property1_violated(input, output) -> bool:

    def scale_COC_output(x):
        return (x - 7.518884) / 373.94992

    def intruder_distant_and_slower(input) -> bool:
        return (
            (input[distanceToIntruder] >= 55947.691) and
            (input[speed] >= 1145) and
            (input[intruderSpeed] <= 60)
        )

    return (
        (valid_input(input) and intruder_distant_and_slower(input)) and
        not (output[clearOfConflict] <= scale_COC_output(1500))
    )


def property2_violated(input, output) -> bool:

    def intruder_distant_and_slower(input) -> bool:
        return (
            (input[distanceToIntruder] >= 55947.691) and
            (input[speed] >= 1145) and
            (input[intruderSpeed] <= 60)
        )
    
    return (
        (valid_input(input) and intruder_distant_and_slower(input)) and
        (torch.argmax(output).item() == clearOfConflict)
    )


def property3_violated(input, output) -> bool:

    def directly_ahead(input) -> bool:
        return (
            (1500 <= input[distanceToIntruder] <= 1800) and
            (-0.06 <= input[angleToIntruder] <= 0.06)
        )

    def moving_towards(input) -> bool:
        return (
            (input[intruderHeading] >= 3.10) and
            (input[speed] >= 980) and
            (input[intruderSpeed] >= 960)
        )

    return (
        (valid_input(input) and directly_ahead(input) and moving_towards(input)) and
        (torch.argmin(output).item() == clearOfConflict)
    )


def property4_violated(input, output) -> bool:

    def moving_away(input) -> bool:
        return (
            (input[intruderHeading] == 0) and
            (1000 <= input[speed]) and
            (700 <= input[intruderSpeed] <= 800)
        )
    
    def directly_ahead(input) -> bool:
        return (
            (1500 <= input[distanceToIntruder] <= 1800) and
            (-0.06 <= input[angleToIntruder] <= 0.06)
        )

    return (
        (valid_input(input) and directly_ahead(input) and moving_away(input)) and
        (torch.argmin(output).item() == clearOfConflict)
    )


def property5_violated(input, output) -> bool:

    def near_and_approaching_from_left(input) -> bool:
        return (
            (250 <= input[distanceToIntruder] <= 400) and
            (0.2 <= input[angleToIntruder] <= 0.4) and
            (-pi <= input[intruderHeading] <= -pi + 0.005) and
            (100 <= input[speed] <= 400) and
            (0 <= input[intruderSpeed] <= 400)
        )

    return (
        (valid_input(input) and near_and_approaching_from_left(input)) and
        not (torch.argmin(output).item() == strongRight)
    )


def property6_violated(input, output) -> bool:

    def intruder_far_away(input) -> bool:
        return (
            (12000 <= input[distanceToIntruder] <= 62000) and
            (- pi <= input[angleToIntruder] <= -0.7 or 0.7 <= input[angleToIntruder] <= pi) and
            (-pi <= input[intruderHeading] <= -pi + 0.005) and
            (100 <= input[speed] <= 1200) and
            (0 <= input[intruderSpeed] <= 1200)
        )

    return (
        (valid_input(input) and intruder_far_away(input)) and
        not ((torch.argmin(output).item() == clearOfConflict))
    )


def property7_violated(input, output) -> bool:

    def large_vertical_separation(input) -> bool:
        return (
            (0 <= input[distanceToIntruder] <= 60760) and
            (-pi <= input[angleToIntruder] <= pi) and
            (100 <= input[speed] <= 1200) and
            (0 <= input[intruderSpeed] <= 1200)
        )

    return (
        (valid_input(input) and large_vertical_separation(input)) and
        ((torch.argmin(output).item() == strongLeft) or (torch.argmin(output).item() == strongRight))
    )


def property8_violated(input, output) -> bool:

    def large_vertical_separation_and_previous_weak_left(input):
        return (
            (0 <= input[distanceToIntruder] <= 60760) and
            (-pi <= input[angleToIntruder] <= -0.75*pi) and
            (-0.1 <= input[intruderHeading] <= 0.1) and
            (600 <= input[speed] <= 1200) and
            (600 <= input[intruderSpeed] <= 1200)
        )

    return (
        (valid_input(input) and large_vertical_separation_and_previous_weak_left(input)) and
        (not (torch.argmin(output).item() == clearOfConflict) and not (torch.argmin(output).item() == weakLeft))
    )


def property9_violated(input, output) -> bool:

    def previous_weak_right_and_nearby_intruder(input) -> bool:
        return (
            (2000 <= input[distanceToIntruder] <= 7000) and
            (-0.4 <= input[angleToIntruder] <= -0.14) and
            (-pi <= input[intruderHeading] <= -pi + 0.01) and
            (100 <= input[speed] <= 150) and
            (0 <= input[intruderSpeed] <= 150)
        )

    return (
        (valid_input(input) and previous_weak_right_and_nearby_intruder(input)) and
        not (torch.argmin(output).item() == strongLeft)
    )


def property10_violated(input, output) -> bool:

    def intruder_far_away2(input) -> bool:
        return (
            (36000 <= input[distanceToIntruder] <= 60760) and
            (0.7 <= input[angleToIntruder] <= pi) and
            (-pi <= input[intruderHeading] <= -pi + 0.01) and
            (900 <= input[speed] <= 1200) and
            (600 <= input[intruderSpeed] <= 1200)
        )

    return (
        (valid_input(input) and intruder_far_away2(input)) and
        not (torch.argmin(output).item() == clearOfConflict)
    )

########################################################################

# Experiments to generate a counter-example for each property

########################################################################

def test_pytorch_search_acasXu_property1() -> None:
    declarations = ["acasXu", "property1"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property1"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property1"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property1_violated(input, output)

        write_search_result("property1", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property1", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property2() -> None:
    declarations = ["acasXu", "property2"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property2"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property2"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property2_violated(input, output)

        write_search_result("property2", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property2", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property3() -> None:
    declarations = ["acasXu", "property3"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property3"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property3"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property3_violated(input, output)

        write_search_result("property3", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property3", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property4() -> None:
    declarations = ["acasXu", "property4"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property4"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property4"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property4_violated(input, output)

        write_search_result("property4", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property4", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property5() -> None:
    declarations = ["acasXu", "property5"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property5"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property5"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property5_violated(input, output)

        write_search_result("property5", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property5", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property6() -> None:
    declarations = ["acasXu", "property6"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property6"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property6"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property6_violated(input, output)

        write_search_result("property6", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property6", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property7() -> None:
    declarations = ["acasXu", "property7"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property7"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property7"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property7_violated(input, output)

        write_search_result("property7", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property7", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property8() -> None:
    declarations = ["acasXu", "property8"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property8"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property8"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property8_violated(input, output)

        write_search_result("property8", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property8", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property9() -> None:
    declarations = ["acasXu", "property9"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property9"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property9"][0]

    if counterexample_found:
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property9_violated(input, output)

        write_search_result("property9", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property9", samples, model, "TIMEOUT")


def test_pytorch_search_acasXu_property10() -> None:
    declarations = ["acasXu", "property10"]
    networks = {"acasXu": model}

    counterexample_found = False
    search_results = None

    start_search = time.perf_counter()
    while time.monotonic() - start_search < 60 * 60:
        search_results = loss_pt.search(
            spec_path,
            logic=VehicleDifferentiableLogic(),
            networks=networks,
            declarations=declarations,
            num_searches=1,
        )
        if len(search_results["property10"][0]) > 0:
            counterexample_found = True
            break
    end_search = time.perf_counter()

    samples = search_results["property10"][0]

    if counterexample_found:
        # If this property evaluates to False, this means we have found
        # exactly one counter-example
        assert len(samples) == 1

        input = samples[0]["x"]

        output = model(normalise(input))

        assert property10_violated(input, output)

        write_search_result("property10", samples, model, (end_search - start_search))
    else:
        assert len(samples) == 0
        write_search_result("property10", samples, model, "TIMEOUT")
