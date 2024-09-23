import sys
import os
import math
import subprocess
from datetime import datetime

from config import *
INSTANCE_DIRECTORY = 'modwheel'

assert os.path.exists(INSTANCE_DIRECTORY)

r = int(sys.argv[1])

os.system(f'mkdir -p {LOCAL_STORAGE}/modwheel_{r}')

def call(command):
#  print('[' + command + ']')
  os.system(command)

for order in sorted(list(range(9, 100, 10)) + list(range(99, 1000, 100))):
  file_base = f'{INSTANCE_DIRECTORY}/modwheel-{order:05d}x{order:05d}#{r:03d}'

  print(f'Considering {file_base} at {datetime.now()}.', flush=True)

  # Run cmr-tu.
  call(f'gunzip -cd {file_base}.sparse.gz | {BUILD_DIRECTORY}/cmr-tu - -i sparse --stats --algo decomposition --time-limit 3600 1> {file_base}-cmrdec.out 2> {file_base}-cmrdec.err')
  if order <= 50:
    call(f'gunzip -cd {file_base}.sparse.gz | {BUILD_DIRECTORY}/cmr-tu - -i sparse --stats --algo eulerian --time-limit 3600 1> {file_base}-cmreuler.out 2> {file_base}-cmreuler.err')
  if order <= 50:
    call(f'gunzip -cd {file_base}.sparse.gz | {BUILD_DIRECTORY}/cmr-tu - -i sparse --stats --algo partition --time-limit 3600 1> {file_base}-cmrpart.out 2> {file_base}-cmrpart.err')
    
  call(f'gunzip -cd {file_base}.sparse.gz | {BUILD_DIRECTORY}/cmr-tu - -i sparse --stats --algo decomposition --time-limit 3600 -N {file_base}-cmrcert.sub 1> {file_base}-cmrcert.out 2> {file_base}-cmrcert.err')

  # Run unimodularity-test.
  call(f'gunzip -cd {file_base}.sparse.gz | {BUILD_DIRECTORY}/cmr-matrix - -i sparse -o dense {LOCAL_STORAGE}/modwheel_{r}/input.dense')
  call(f'{UNIMOD_DIRECTORY}/unimodularity-test {LOCAL_STORAGE}/modwheel_{r}/input.dense -s 2> /dev/null | egrep \'^[ 0-9-]*$\' 1> {LOCAL_STORAGE}/modwheel_{r}/signed.dense')
  call(f'{UNIMOD_DIRECTORY}/unimodularity-test {LOCAL_STORAGE}/modwheel_{r}/signed.dense -t -v 1> {file_base}-unimod.out 2> {file_base}-unimod.err')
  if order <= 199:
    call(f'{UNIMOD_DIRECTORY}/unimodularity-test {LOCAL_STORAGE}/modwheel_{r}/signed.dense -t -v -c 1> {file_base}-unimodcert.out 2> {file_base}-unimodcert.err')

os.system(f'rm -r {LOCAL_STORAGE}/modwheel_{r}')

