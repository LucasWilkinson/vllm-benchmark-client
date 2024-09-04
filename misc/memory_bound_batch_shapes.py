import os
import sys
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir + "/../")

from misc.weight_shapes import WEIGHT_SHAPES

TFLOPS_FP16 = {
    "a100-sxm": 312,
}

GBS = {
    "a100-sxm": 2039,
}

machine = "a100-sxm"
machine_balance_fp16 = (TFLOPS_FP16[machine] * 1e12) / (GBS[machine] * 1024**3)
print(f"Machine: {machine}")
print(f"Machine Balance: {machine_balance_fp16:.2f} TFLOPS/GB/s")
for model_name, shapes in WEIGHT_SHAPES.items():
    print(model_name)
    for shape in shapes:
        # arithmetic intensity = m * k * n / (2 * ((m * n) + (m * k) + (k * n)))
        #  (2 in the denominator is for 2 bytes per float16 value)
        # solve for m by setting arithmetic intensity to machine balance
        # let ai = arithmetic intensity
        # m = -1 * (2 * ai * k * n) / (2 * ai * (k + n) - k * n)
        k, n = shape[0]
        ai = machine_balance_fp16
        m = -1 * (2 * ai * k * n) / (2 * ai * (k + n) - k * n)
        print(f"   {str(shape[0]):<13},  memory/compute bound m boundry = {m}")