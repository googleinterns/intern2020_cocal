import os, sys
import argparse

def to_tree_namespace(args):
    """
    handle only depth-1 tree
    """
    for key, val in vars(args).copy().items():
        if "." in key:
            delattr(args, key)
            group, sub_key = key.split(".", 2)
            if not hasattr(args, group):
                setattr(args, group, argparse.Namespace())
            setattr(getattr(args, group), sub_key, val)
    return args


def print_args(args, args_str="args", n_tap_str=1):
    print("\t"*(n_tap_str-1)+args_str + ":" )
    for key, val in vars(args).items():
        if "Namespace" in str(type(val)):
            print_args(val, args_str=key, n_tap_str=n_tap_str+1)
        else:
            print("\t"*n_tap_str + key + ":", val)
    print()
