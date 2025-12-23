import time
import pickle


def timed(f, suffix=""):
    t0 = time.time()
    res = f()
    dt = time.time() - t0
    print(f"Elapsed{suffix}: {dt:.4f}s")
    return res


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)