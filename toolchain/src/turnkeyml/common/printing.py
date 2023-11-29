import os
import re
import enum
import sys
import math


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def log(txt, c=Colors.ENDC, end="", is_error=False):
    logn(txt, c=c, end=end, is_error=is_error)


def logn(txt, c=Colors.ENDC, end="\n", is_error=False):
    file = sys.stderr if is_error else sys.stdout
    print(c + txt + Colors.ENDC, end=end, flush=True, file=file)


class LogType(enum.Enum):
    ERROR = "Error:"
    SUCCESS = "Woohoo!"
    WARNING = "Warning:"
    INFO = "Info:"


def clean_print(type: LogType, msg):
    # Replace path to user’s home directory by a tilde symbol (~)
    home_directory = os.path.expanduser("~")
    home_directory_escaped = re.escape(home_directory)
    msg = re.sub(home_directory_escaped, "~", msg)

    # Split message into list, remove leading spaces and line breaks
    msg = msg.split("\n")
    msg = [line.lstrip() for line in msg]
    while msg[0] == "" and len(msg) > 1:
        msg.pop(0)

    # Print message
    indentation = len(type.value) + 1
    if type == LogType.ERROR:
        log(f"\n{type.value} ".rjust(indentation), c=Colors.FAIL, is_error=True)
    elif type == LogType.SUCCESS:
        log(f"\n{type.value} ".rjust(indentation), c=Colors.OKGREEN)
    elif type == LogType.WARNING:
        log(f"\n{type.value} ".rjust(indentation), c=Colors.WARNING)
    elif type == LogType.INFO:
        log(f"\n{type.value} ".rjust(indentation), c=Colors.OKCYAN)

    is_error = type == LogType.ERROR
    for line_idx, line in enumerate(msg):
        if line_idx != 0:
            log(" " * indentation)
        s_line = line.split("**")
        for idx, l in enumerate(s_line):
            c = Colors.ENDC if idx % 2 == 0 else Colors.BOLD
            if idx != len(s_line) - 1:
                log(l, c=c, is_error=is_error)
            else:
                logn(l, c=c, is_error=is_error)


def log_error(msg):
    clean_print(LogType.ERROR, str(msg))
    # ASCII art credit:
    # https://textart4u.blogspot.com/2014/05/the-fail-whale-ascii-art-code.html
    logn(
        """\n▄██████████████▄▐█▄▄▄▄█▌
██████▌▄▌▄▐▐▌███▌▀▀██▀▀
████▄█▌▄▌▄▐▐▌▀███▄▄█▌
▄▄▄▄▄██████████████\n\n""",
        is_error=True,
    )


def log_success(msg):
    clean_print(LogType.SUCCESS, msg)


def log_warning(msg):
    clean_print(LogType.WARNING, msg)


def log_info(msg):
    clean_print(LogType.INFO, msg)


def list_table(list, padding=25, num_cols=4):
    lines_per_column = int(math.ceil(len(list) / num_cols))
    for i in range(lines_per_column):
        for col in range(num_cols):
            if i + col * lines_per_column < len(list):
                print(
                    list[i + col * lines_per_column].ljust(padding),
                    end="",
                )
        print("\n\t", end="")
