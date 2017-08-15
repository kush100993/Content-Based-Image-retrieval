import os
import pickle


# Pickle load a file. Returns None if file does not exist
def load(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except OSError:
        return None


# Pickle dump a file
def dump(obj, filepath):
    with open(filepath, 'wb+') as f:
        return pickle.dump(obj, f)


# Delete a file
def delete(filepath):
    try:
        os.remove(filepath)
    except IOError:
        pass
