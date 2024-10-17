import pandas as pd
import numpy as np
import os
import shutil
import warnings
warnings.filterwarnings("ignore")


gigahorse_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/gigahorse-toolchain/"
bytecode_dir = os.path.dirname(os.path.abspath(__file__)) + "/bytecode/"


def run_gigahorse(bytecode_path):
    os.system("python3 {} -C {} {}".format(
        gigahorse_dir + "gigahorse.py",
        gigahorse_dir + "clients/visualizeout.py",
        bytecode_path
    ))


if __name__ == "__main__":
    for k, bytecode_file in enumerate(os.listdir(bytecode_dir)):
        addr = bytecode_file[:-4]
        run_gigahorse(bytecode_dir + bytecode_file)
        print("{} {}\n".format(k, addr))