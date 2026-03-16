import sys
print("Python:", sys.executable)
try:
    import torch
    print("torch OK:", torch.__version__)
except ImportError as e:
    print("torch FAIL:", e)
try:
    import numpy
    print("numpy OK:", numpy.__version__)
except ImportError as e:
    print("numpy FAIL:", e)
try:
    import yaml
    print("yaml OK")
except ImportError as e:
    print("yaml FAIL:", e)
