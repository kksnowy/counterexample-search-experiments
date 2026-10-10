from pathlib import Path
from datetime import datetime
import vehicle_lang as vcl
import multiprocessing
import time

ONNX_MODELS_DIR = Path(__file__).parent.parent / "models" / "onnx"

PROPERTY = "property1" # Replace with desired property

def verify():
	result = vcl.verify(
		specification = "acasXu.vcl",
		networks={"acasXu": ONNX_MODELS_DIR / "acasXu_1_7.onnx"},
		solver="Marabou",
		properties=[PROPERTY]
	)
	return result

def run_verify(queue):
	result = verify()
	queue.put(result)

if __name__ == "__main__":
	queue = multiprocessing.Queue()

	process = multiprocessing.Process(target=run_verify, args=(queue,))

	start = time.perf_counter()
	process.start()
	process.join(timeout=60 * 60)
	end = time.perf_counter()

	if process.is_alive():
		process.terminate()
		process.join()
		result = "TIMEOUT"
	else:
		result = queue.get()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_filename = f"results/verification/acasXu_verify_{PROPERTY}_{timestamp}.txt"
	with open(results_filename, "w") as file:
		file.write("Output: " + str(result) + "\n")
		file.write("Time: " + str(end - start))