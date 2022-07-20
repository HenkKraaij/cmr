# Instance Generators # {#generators}

The library comes with several instance generators.
However, they are disabled for compilation by default, and can be enabled by adding

    -DGENERATORS=on

to the cmake call.
In the following we provide details for each of these generators.

## Graphic Matrices ##

The executable `cmr-generate-graphic` creates a random \f$ m \f$-by-\f$n\f$ matrix that is graphic.
To this end, it first constructs a spanning tree with \f$ m+1 \f$ nodes by subsequently connecting each new leaf to one of the previously created nodes.
The new edge then indexes a row of the matrix.
Second, \f$ n \f$ random pairs of distinct nodes are generated that are then connected by a new edge indexed by a column.
The matrix is the representation matrix of the final graph with respect to the spanning tree.
It can be called as follows, where \f$ m \f$ equals ROWS and \f$ n \f$ equals COLS.

    ./cmr-generate-graphic [OPTIONS] ROWS COLS

Options:
  - `-b NUM`    Benchmarks the recognition algorithm for the created matrix with NUM repetitions.
  - `-o FORMAT` Format of output FILE; default: `dense`.

Formats for matrices are \ref dense-matrix, \ref sparse-matrix.

## Network Matrices ##

The executable `cmr-generate-network` creates a random \f$ m \f$-by-\f$n\f$ network matrix.
To this end, it first constructs a random graphic matrix (see above) and then modifies its signs via \ref camion.
It can be called as follows, where \f$ m \f$ equals ROWS and \f$ n \f$ equals COLS.

    ./cmr-generate-network [OPTIONS] ROWS COLS

Options:
  - `-b NUM`    Benchmarks the recognition algorithm for the created matrix with NUM repetitions.
  - `-o FORMAT` Format of output FILE; default: `dense`.

Formats for matrices are \ref dense-matrix, \ref sparse-matrix.

## Random Matrices ##

The executable `cmr-generate-random` creates a random \f$ m \f$-by-\f$n\f$ binary matrix whose entries are chosen uniformly at random with a given probability \f$ p \f$.
It can be called as follows, where \f$ m \f$ equals ROWS and \f$ n \f$ equals COLS.

    ./cmr-generate-random [OPTIONS] ROWS COLS p

Options:
  - `-o FORMAT` Format of output FILE; default: `dense`.

Formats for matrices are \ref dense-matrix, \ref sparse-matrix.

## Random Perturbations ##

The executable `cmr-perturb-random` modifies a matrix by applying a specified number of random perturbations of different types.
It can be called as follows.

    ./cmr-perturb-random [OPTIONS] MATRIX

Options:
  - `-i FORMAT` Format of MATRIX file, among {dense, sparse}; default: dense.
  - `-o FORMAT` Format of output, among {dense, sparse}; default: same as input.
  - `-0 NUM`    Turn NUM randomly chosen nonzero entries to 0s.
  - `-1 NUM`    Turn NUM randomly chosen zero entries into 1s.
  - `--1 NUM`   Turn NUM randomly chosen zero entries into -1s.
  - `-b NUM`    Flip NUM randomly chosen entries over the binary field.
  - `-t NUM`    Flip NUM randomly chosen entries over the ternary field.

If MATRIX is `-`, then the matrix will be read from stdin.
