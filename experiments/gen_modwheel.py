import sys
import os
import math
import subprocess

BUILD_DIRECTORY = '../build-release'
OUTPUT_DIRECTORY = 'modwheel'

try:
  num_repetitions = int(sys.argv[1])
except:
  print(f'Usage: {sys.argv[0]} NUM-REPETITIONS')
  sys.exit(1)

if not os.path.exists(OUTPUT_DIRECTORY):
  os.mkdir(OUTPUT_DIRECTORY)

for order in sorted(list(range(3, 50, 2)) + list(range(99, 1000, 50))):
  for r in range(1, num_repetitions+1):
    file_base = f'{OUTPUT_DIRECTORY}/modwheel-{order:05d}x{order:05d}#{r:03d}'
    os.system(f'{BUILD_DIRECTORY}/cmr-generate-wheel -01 {order} -o sparse | {BUILD_DIRECTORY}/cmr-matrix -i sparse - -r -R2 {order//2+1} -o sparse - | gzip > {file_base}.sparse.gz')
