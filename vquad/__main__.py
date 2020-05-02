# Copyright 2018 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import sys
import traceback
from . import benchmark

# Make sure that whenever there is an exception it is printed and an
# appropriate exit code is set.
rc = 1
try:
    benchmark.main()
    rc = 0
except (Exception, KeyboardInterrupt):
    traceback.print_exc()
finally:
    sys.exit(rc)
