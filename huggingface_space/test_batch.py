import pandas as pd
from app import process_batch_prediction

print("Testing with None:")
res = process_batch_prediction(None)
print("Result with None:", res)

print("\nTesting with a mock string path:")
try:
    with open("test_mock.csv", "w", encoding="utf-8") as f:
        f.write("review\ngood\nbad\n")
    res = process_batch_prediction("test_mock.csv")
    print("Result with string path:", res)
except Exception as e:
    print("Exception outside:", type(e), e)
