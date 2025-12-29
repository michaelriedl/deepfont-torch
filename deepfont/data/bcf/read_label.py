import numpy as np


def read_label(label_file: str) -> np.ndarray:
    """Reads a binary label file and returns its contents as a NumPy array.

    Opens a binary label file containing unsigned 32-bit integer labels and loads
    them into a NumPy array. This function reads the entire file contents into memory
    at once, so it's most suitable for label files that fit comfortably in RAM.

    The label file is expected to be in raw binary format with labels stored as
    contiguous uint32 values (4 bytes each, little-endian). No header or metadata
    is expected in the file.

    Args:
        label_file: The file path to the binary label file. Must be a valid path
            to an existing file containing uint32 labels in binary format.

    Returns:
        A 1-dimensional NumPy array of dtype uint32 containing all labels from
        the file. The array length equals the number of 4-byte integers in the file.

    Raises:
        FileNotFoundError: If the specified label file does not exist.
        IOError: If there's an error opening or reading the file.
        MemoryError: If the file is too large to fit in available memory.
    """
    with open(label_file, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint32)

    return labels
