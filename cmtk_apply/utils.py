import os
import pathlib
import platform

_search_path = os.environ["PATH"]
_search_path = [i for i in _search_path.split(os.pathsep) if len(i) > 0]
_search_path += [
    "~/bin",
    "/usr/lib/cmtk/bin/",
    "/usr/local/lib/cmtk/bin",
    "/usr/local/bin",
    "/opt/local/bin",
    "/opt/local/lib/cmtk/bin/",
    "/Applications/IGSRegistrationTools/bin",
]

if platform.system() == "Windows":
    _search_path += [
        r"C:\cygwin64\usr\local\lib\cmtk\bin",
        r"C:\Program Files\CMTK-3.3\CMTK\lib\cmtk\bin",
    ]


def find_streamxform() -> str:
    """Find directory with binaries."""
    for path in _search_path:
        path = pathlib.Path(path).absolute()
        if not path.is_dir():
            continue

        try:
            return next(path.glob("streamxform")).resolve()
        except StopIteration:
            continue
        except BaseException:
            raise
    return None
