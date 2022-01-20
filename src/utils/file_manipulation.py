import errno
import os


def silentremove(filename: str):
    """
    Remove a file without raising an error if the file does not exist.
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def get_environ(variable_name: str):
    """
    Get the value of a variable from the environment.
    """
    variable = os.environ.get(variable_name)
    if variable is None:
        raise ValueError(f"The environment variable {variable_name} is not defined.")
    return variable
