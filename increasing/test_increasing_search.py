from typing import Any
from datetime import datetime

import csv
import time
import torch

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import VehicleDifferentiableLogic


def write_increasing_search_results(
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
                x_value = samples[0]["x"]

                input_value = x_value.unsqueeze(0)
                input.append(input_value.tolist())

                output_value = model(input_value)
                output.append(output_value.tolist())

                counterexample_found.append(True)
            else:
                input.append(None)
                output.append(None)
                counterexample_found.append(False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/increasing_{timestamp}.csv"

    with open(results_filename, "w", newline="") as results_file:
        writer = csv.writer(results_file)

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


def test_pytorch_search_increasing() -> None:
    spec_path = "increasing.vcl"
    declarations = ["f", "increasing"]

    # A network that calculates y = |x - 0.5|
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1, bias=False)
    )
    with torch.no_grad():
        model[0].weight[:] = torch.tensor([[1.0], [-1.0]])
        model[0].bias[:] = torch.tensor([-0.5, 0.5])
        model[2].weight[:] = torch.tensor([1.0, 1.0])

    networks = {"f": model}
    
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
                x = samples[0]["x"].unsqueeze(0)

                output = model(x)

                # Check that the counter-example actually violates the increasing property
                assert 0 < x < 1
                assert not (x <= output)
            else:
                # If no counter-example was found, a search's list of results is empty
                assert len(samples) == 0

    write_increasing_search_results(search_results, model, end - start)
