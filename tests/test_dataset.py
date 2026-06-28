import pandas as pd
from gpusitter.detection.dataset import build_xy, time_split

def test_build_xy_shapes_and_labels():
    hist = [
        {"gpu_num": 8, "duration": 100, "type": "train", "state": "COMPLETED"},
        {"gpu_num": 16, "duration": 200, "type": "eval", "state": "NODE_FAIL"},
    ]
    X, y = build_xy(hist, ["gpu_num", "duration", "type"])
    assert y == [0, 1]
    assert list(X["gpu_num"]) == [8, 16]
    assert "type_train" in X.columns and "type_eval" in X.columns

def test_time_split_respects_order():
    X = pd.DataFrame({"a": list(range(1, 11))})
    y = [0, 1] * 5
    Xtr, ytr, Xval, yval = time_split(X, y, val_frac=0.3)
    assert len(ytr) == 7 and len(yval) == 3
    assert list(Xval["a"]) == [8, 9, 10]
