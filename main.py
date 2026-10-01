#!/usr/bin/env python3

import random
import numpy as np
import codes

RANDOM_SEED = 0

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    codes.run_search()

if __name__ == "__main__":
    main()