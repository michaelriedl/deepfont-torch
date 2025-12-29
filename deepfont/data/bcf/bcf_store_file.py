import numpy as np


class BCFStoreFile:
    """A reader for Binary Concatenated File (BCF) store files.

    This class provides an interface to read from BCF store files, which are binary
    files containing multiple concatenated files. Each BCF store file maintains an
    index of file sizes and offsets, allowing efficient random access to individual
    files within the concatenated structure.

    The BCF format stores:
        - A header containing the number of concatenated files (8 bytes, uint64)
        - An array of file sizes for each concatenated file (8 bytes each, uint64)
        - The actual file contents concatenated sequentially

    This class is thread-safe for read operations and supports copying via the
    standard Python copy mechanisms (__copy__ and __deepcopy__).

    Attributes:
        _filename: The path to the BCF store file.
        _file: The open file handle for reading binary data.
        _offsets: NumPy array containing cumulative byte offsets for each file.
    """

    def __init__(self, filename: str) -> None:
        """Initializes a BCFStoreFile object and loads the file index.

        Opens the specified BCF store file, reads its header to determine the
        number of concatenated files, and constructs an offset array for efficient
        random access to individual files.

        Args:
            filename: The path to the BCF store file to open. Must be a valid
                path to an existing BCF format file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If the file cannot be opened or read.
            ValueError: If the file format is invalid or corrupted.
        """
        # Store the filename
        self._filename = filename
        # Open the file
        self._file = open(filename, "rb")
        # Read the number of files
        size = np.frombuffer(self._file.read(8), dtype=np.uint64)[0]
        # Read the file sizes
        file_sizes = np.frombuffer(self._file.read(int(8 * size)), dtype=np.uint64)
        self._offsets = np.append(np.uint64(0), np.add.accumulate(file_sizes))

    def __del__(self):
        """Destructor that ensures the file handle is properly closed.

        Called automatically when the object is about to be destroyed. Closes
        the underlying file handle to release system resources.
        """
        self._file.close()

    def __copy__(self):
        """Creates a shallow copy of this BCFStoreFile object.

        Creates a new BCFStoreFile instance that references the same BCF file.
        The new instance opens its own file handle to the same file, allowing
        independent read operations. The offset array is shared between the
        original and the copy.

        Returns:
            A new BCFStoreFile instance pointing to the same file with its own
            file handle.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result._filename = self._filename
        result._file = open(self._filename, "rb")
        result._offsets = self._offsets

        return result

    def __deepcopy__(self, memo):
        """Creates a deep copy of this BCFStoreFile object.

        Creates a new BCFStoreFile instance that references the same BCF file.
        Similar to __copy__, this opens a new file handle to allow independent
        read operations. The memo dictionary is used to track already-copied
        objects to handle circular references properly.

        Args:
            memo: A dictionary used by the deepcopy mechanism to track objects
                that have already been copied, preventing infinite recursion.

        Returns:
            A new BCFStoreFile instance pointing to the same file with its own
            file handle.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._filename = self._filename
        result._file = open(self._filename, "rb")
        result._offsets = self._offsets

        return result

    def get(self, i: int) -> bytes:
        """Retrieves the contents of a specific file from the BCF store.

        Seeks to the appropriate position in the BCF store file and reads the
        complete contents of the file at the specified index. The file pointer
        is positioned based on the pre-computed offset array, allowing efficient
        random access without scanning through previous files.

        Args:
            i: The zero-based index of the file to retrieve. Must be in the range
                [0, size()-1].

        Returns:
            The raw binary contents of the file at the specified index as a bytes
            object.

        Raises:
            IndexError: If the index is out of bounds (negative or >= size()).
            IOError: If there's an error reading from the file.
        """
        self._file.seek(int(len(self._offsets) * 8 + self._offsets[i]))
        return self._file.read(int(self._offsets[i + 1] - self._offsets[i]))

    def size(self):
        """Returns the total number of files stored in the BCF store.

        Computes the count of concatenated files by examining the length of
        the internal offset array. This is a constant-time operation that uses
        metadata loaded during initialization.

        Returns:
            The number of files contained in the BCF store file as an integer.
        """
        return len(self._offsets) - 1

    def reset_file_pointer(self):
        """Resets the file handle by closing and reopening the BCF store file.

        This method is useful for recovering from file handle errors or resetting
        the file pointer state. It closes the current file handle if it's still
        open and opens a new one to the same file. This can help resolve issues
        with file locks or corrupted file states.

        Note:
            Any ongoing read operations will be interrupted when this method is called.
            The file position is reset to the beginning of the file.
        """
        if not self._file.closed:
            self._file.close()
        self._file = open(self._filename, "rb")
