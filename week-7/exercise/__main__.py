import sys
from .numpy import main as numpy_main
from .sklearn import main as sklearn_main

if __name__ == "__main__":
    match sys.argv[1]:
        case "numpy":
            numpy_main()
        case "sklearn":
            sklearn_main()
        case _:
            raise RuntimeError(
                f"Invalid argument: {sys.argv[1]}, valid values: numpy, sklearn"
            )
