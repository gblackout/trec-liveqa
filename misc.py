import numpy as np


def linesep(q, sep='='):
    tmp = 80 - len(q)
    slen = tmp / 2 + (tmp % 2 == 0)

    print '\n', sep*slen, q, sep*slen, '\n'