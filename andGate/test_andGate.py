from typing import Any
from datetime import datetime

import csv
import time
import torch

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import VehicleDifferentiableLogic


def write_andGate_search_results(
    search_results: dict[str, Any], 
    model: Any, 
    time: float
) -> None:

    result_id = []
    counterexample_found = []
    input = []
    output = []
    
    result_count = 0

    for _, results in search_results.items():
        for samples in results:
            result_count += 1
            result_id.append(result_count)

            if len(samples) > 0:
                x1_value = samples[0]["x1"]
                x2_value = samples[0]["x2"]

                input_value = torch.stack([x1_value, x2_value])
                input.append(input_value.tolist())

                output_value = model(input_value)
                output.append(output_value.tolist())

                counterexample_found.append(True)
            else:
                input.append(None)
                output.append(None)
                counterexample_found.append(False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_filename = f"results/andGate_{timestamp}.csv"

    with open(sample_filename, "w", newline="") as sample_file:
        writer = csv.writer(sample_file)

        # Write column names
        writer.writerow([
            "result_id",
            "counterexample_found",
            "input",
            "output",
            "time" # Note that this is the time for ALL searches
        ])

        for i in range(len(result_id)):
            writer.writerow([
                result_id[i],
                counterexample_found[i],
                input[i],
                output[i],
                time
            ])


def test_pytorch_search_andGate() -> None:
    spec_path = "andGate.vcl"
    declarations = ["andGate", "andGateCorrect"]

    # A network which takes a tensor [x, y] and calculates x - y
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, -1.0]])

    networks = {"andGate": model}

    start = time.perf_counter()
    search_results = loss_pt.search(
        spec_path,
        logic=VehicleDifferentiableLogic(),
        declarations=declarations,
        networks=networks,
        num_searches=4,
    )
    end = time.perf_counter()
    
    for _, results in search_results.items():
        for samples in results:

            if len(samples) > 0:
                # This property is compiled into a LeafDisjunctionNegated so a counter-example
                # consists of a witness for exactly 1 leaf disjunct
                assert len(samples) == 1
                x1 = samples[0]["x1"]
                x2 = samples[0]["x2"]

                inputs = torch.stack([x1, x2])
                output = model(inputs)

                # Check that the counter-example actually violates the andGateCorrect property
                assert (0 <= x1 <= 1) and (0 <= x2 <= 2)
                assert (
                    (x1 >= 0.5 and x2 >= 0.5 and not output >= 0.5)
                    or (x1 >= 0.5 and x2 <= 0.5 and not output <= 0.5)
                    or (x1 <= 0.5 and x2 >= 0.5 and not output <= 0.5)
                    or (x1 <= 0.5 and x2 <= 0.5 and not output <= 0.5)
                )
            else:
                # If no counter-example was found, a search's list of results is empty
                assert len(samples) == 0

    write_andGate_search_results(search_results, model, end - start)
