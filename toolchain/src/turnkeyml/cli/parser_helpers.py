from typing import Dict, List, Optional
import turnkeyml.common.exceptions as exp


def decode_args(args: Optional[List[str]]) -> Dict:
    """
    Decode CLI arguments that contain the format
    ["arg1::[value1,value2]", "arg2::value1", "flag_arg"]
    """
    arg_dict = dict()
    if args is None:
        return arg_dict
    for arg in args:
        split_arg = arg.split("::")
        if len(split_arg) == 1:
            # Arguments without values (flags)
            arg_dict[split_arg[0]] = True
        elif len(split_arg) == 2:
            if split_arg[1].startswith("[") and split_arg[1].endswith("]"):
                # Arguments with lists of values
                arg_dict[split_arg[0]] = split_arg[1][1:-1].split(",")
            else:
                # Arguments with single value
                arg_dict[split_arg[0]] = split_arg[1]
        else:
            raise exp.ArgError(
                (
                    f"Coudn't decode rt-arg argument {arg}. The full set "
                    f"of args received was: {args}. Please make sure "
                    "that your rt-arg arguments adhere to the format "
                    "'arg1::value1,value2 arg2::value1 flag_arg'"
                )
            )
    return arg_dict


def encode_args(args: Dict[str, Optional[List]]) -> List[str]:
    """
    Encode Dict into CLI arguments in the format
    ["arg1::[value1,value2]", "arg2::value1", "flag_arg"]
    """
    if args:
        encoded_dict = []
        for key, value in args.items():
            if value is True:
                encoded_dict.append(key)
            elif not isinstance(value, list):
                encoded_dict.append(f"{key}::{value}")
            else:
                encoded_dict.append(f"{key}::[{','.join(value)}]")
        return encoded_dict
    else:
        return []
