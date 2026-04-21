import os
import inspect

def get_caller_dir(file):
    caller_file = None
    for frame_info in inspect.stack()[1:]:
        fname = os.path.abspath(frame_info.filename)
        if fname != os.path.abspath(file):
            caller_file = fname
            break
    if caller_file is None:
        caller_file = os.path.abspath(inspect.stack()[1].filename)
    return os.path.dirname(caller_file)
