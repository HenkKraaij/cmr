#ifndef CMR_SEPARATION_H
#define CMR_SEPARATION_H

#include <cmr/env.h>
#include <cmr/matrix.h>

#include <assert.h>
#include <stdint.h>

/**
 * \file separation.h
 *
 * \author Matthias Walter
 *
 * \brief Data structures for k-separations.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  unsigned char* rowsToPart;        /**< \brief Indicates to which block each row belongs. Values above 1 are ignored. */
  unsigned char* columnsToPart;     /**< \brief Indicates to which block each column belongs. Values above 1 are ignored. */
  size_t numRows[2];                /**< \brief Indicates the number of rows of each part. */
  size_t numColumns[2];             /**< \brief Indicates the number of columns of each part. */
  size_t* rows[2];                  /**< \brief Array of sorted rows for each part. */
  size_t* columns[2];               /**< \brief Array of sorted columns for each part. */
  size_t extraRows0[2];             /**< \brief Extra rows for part 0 or \c SIZE_MAX if bottom-left rank is lower. */
  size_t extraColumns1[2];          /**< \brief Extra columns for part 1 or \c SIZE_MAX if bottom-left rank is lower. */
  size_t extraRows1[2];             /**< \brief Extra rows for part 1 or \c SIZE_MAX if top-right rank is lower. */
  size_t extraColumns0[2];          /**< \brief Extra columns for part 0 or \c SIZE_MAX if top-right rank is lower. */
  unsigned char* indicatorMemory;   /**< \brief Memory for \ref rowsToPart and \ref columnsToPart. */
  size_t* elementMemory;            /**< \brief Memory for \ref rows and \ref columns. */
} CMR_SEPA;

/**
 * \brief Creates a separation.
 *
 * Only the memory is allocated. The usualy way to initialize it is to fill the arrays \ref rowsToPart and
 * \ref columnsToPart and then call \ref CMRsepaInitialize.
 */

CMR_EXPORT
CMR_ERROR CMRsepaCreate(
  CMR* cmr,           /**< \ref CMR environment. */
  size_t numRows,     /**< Number of rows. */
  size_t numColumns,  /**< Number of columns. */
  CMR_SEPA** psepa    /**< Pointer for storing the created separation. */
);

/**
 * \brief Initializes a separation.
 *
 * Assumes that \p separation was created via \ref CMRsepaCreate and that all entries of \ref rowsToPart and
 * \ref columnsToPart are set to either 0 or 1.
 */

CMR_EXPORT
CMR_ERROR CMRsepaInitialize(
  CMR* cmr,                   /**< \ref CMR environment. */
  CMR_SEPA* sepa,             /**< Already created separation. */
  size_t firstExtraRow0,      /**< First extra row for part 0 or \c SIZE_MAX if bottom-left rank is 0. */
  size_t firstExtraColumn1,   /**< First extra column for part 1 or \c SIZE_MAX if bottom-left rank is 0. */
  size_t firstExtraRow1,      /**< First extra row for part 1 or \c SIZE_MAX if top-right rank is 0. */
  size_t firstExtraColumn0,   /**< First extra column for part 0 or \c SIZE_MAX if top-right rank is 0. */
  size_t secondExtraRow0,     /**< Second extra row for part 0 or \c SIZE_MAX if bottom-left rank is at most 1. */
  size_t secondExtraColumn1,  /**< Second extra column for part 1 or \c SIZE_MAX if bottom-left rank is at most 1. */
  size_t secondExtraRow1,     /**< Second extra row for part 1 or \c SIZE_MAX if top-right rank is at most 1. */
  size_t secondExtraColumn0   /**< Second extra column for part 0 or \c SIZE_MAX if top-right rank is at most 1. */
);

/**
 * \brief Frees a separation.
 */

CMR_EXPORT
CMR_ERROR CMRsepaFree(
  CMR* cmr,         /**< \ref CMR environment. */
  CMR_SEPA** psepa  /**< Pointer to separation. */
);

/**
 * \brief Returns rank of bottom-left submatrix.
 */

static inline
unsigned char CMRsepaRankBottomLeft(
  CMR_SEPA* sepa  /**< Separation. */
)
{
  assert(sepa);

  return sepa->extraRows0[0] == SIZE_MAX ? 0
    : (sepa->extraRows0[1] == SIZE_MAX ? 1 : 2);
}

/**
 * \brief Returns rank of top-right submatrix.
 */

static inline
unsigned char CMRsepaRankTopRight(
  CMR_SEPA* sepa  /**< Separation. */
)
{
  return sepa->extraRows1[0] == SIZE_MAX ? 0
    : (sepa->extraRows1[1] == SIZE_MAX ? 1 : 2);
}

/**
 * \brief Returns rank sum of bottom-left and top-right submatrices.
 */

static inline
unsigned char CMRsepaRank(
  CMR_SEPA* sepa  /**< Separation. */
)
{
  return CMRsepaRankBottomLeft(sepa) + CMRsepaRankTopRight(sepa);
}

/**
 * \brief Checks for a given matrix whether the binary k-separation is also a ternary one.
 *
 * Checks, for a ternary input matrix \f$ M \f$ and a k-separation (\f$ k \in \{1,2,3\} \f$) of the (binary) support
 * matrix of \f$ M \f$, whether it is also a k-separation of \f$ M \f$ itself. The result is stored in \p *pisTernary.
 *
 * If the check fails, a certifying submatrix is returned.
 */

CMR_EXPORT
CMR_ERROR CMRsepaCheckTernary(
  CMR* cmr,               /**< \ref CMR environment. */
  CMR_SEPA* sepa,         /**< Separation. */
  CMR_CHRMAT* matrix,     /**< Matrix. */
  bool* pisTernary,       /**< Pointer for storing whether the check passed. */
  CMR_SUBMAT** psubmatrix /**< Pointer for storing a violator submatrix (may be \c NULL). */
);

#ifdef __cplusplus
}
#endif

#endif /* CMR_SEPARATION_H */
