#ifndef CMR_GRAPHIC_H
#define CMR_GRAPHIC_H

#include <cmr/env.h>
#include <cmr/element.h>
#include <cmr/matrix.h>
#include <cmr/graph.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup Graphic Representation Matrices of Graphs
 *
 * @{
 */

/**
 * \brief Computes the binary representation matrix for a given graph.
 *
 * Let \f$ G = (V,E) \f$ be an undirected graph with nodes \f$ V \f$ and edges \f$ E \f$.
 * Let \f$ T \subseteq E \f$ be a spanning forest of \f$ G \f$.
 * The **binary representation matrix** \f$ M := M(G,T) \f$ is a matrix \f$ M \in \{0,1\}^{T \times (E \setminus T)} \f$
 * with \f$ M_{e,f} = 1 \iff e \f$ belongs to the unique cycle in \f$ T \cup \{f\} \f$.
 *
 * Computes \f$ M(G,T) \f$ for given \f$ G \f$ and spanning forest \f$ T \f$ given by \p forestEdges.
 * If \p forestEdges is \c NULL, \f$ T \f$ is some spanning forest \f$ T \f$ of \f$ G \f$ is computed.
 * The ordering of the columns can be specified via \p coforestEdges.
 *
 * \note The function computes a representation matrix of \f$ G \f$ regardless of whether \p forestEdges is a correct
 * spanning tree. This is indicated via *\p pisCorrectForest.
 */

CMR_EXPORT
CMR_ERROR CMRcomputeGraphBinaryRepresentationMatrix(
  CMR* cmr,                       /**< \ref CMR environment. */
  CMR_GRAPH* graph,              /**< Graph \f$ G \f$. */
  CMR_CHRMAT** pmatrix,          /**< Pointer for storing \f$ M \f$ (may be \c NULL). */
  CMR_CHRMAT** ptranspose,       /**< Pointer for storing \f$ M^{\mathsf{T}} \f$ (may be \c NULL). */
  int numForestEdges,           /**< Length of \p forestEdges (0 if \c forestEdges is \c NULL). */
  CMR_GRAPH_EDGE* forestEdges,   /**< If not \c NULL, spanning forest edges as rows in this order. */
  int numCoforestEdges,         /**< Length of \p coforestEdges (0 if \c coforestEdges is \c NULL). */
  CMR_GRAPH_EDGE* coforestEdges, /**< If not \c NULL, complement of forest edges as columns in this order. */
  bool* pisCorrectForest        /**< Pointer for storing whether \c forestEdges is a spanning forest of \f$ G \f$ (may be \c NULL). */
);

/**
 * \brief Computes the ternary representation matrix for a given graph.
 *
 * Let \f$ G = (V,E) \f$ be a directed graph with nodes \f$ V \f$ and edges \f$ E \f$.
 * Let \f$ T \subseteq E \f$ be a spanning forest of \f$ G \f$, directed arbitrarily.
 * The **ternary representation matrix** \f$ M := M(G,T) \f$ is a matrix
 * \f$ M \in \{-1,0,1\}^{T \times (E \setminus T)} \f$ with \f$ M_{e,\{u,v\}} = +1 \f$ (resp.\ \f$ M_{e,\{u,v\}} \f$ if
 * \f$ e \f$ is a forward edge (resp.\ backward edge) on the unique \f$ s \f$-\f$ t \f$-path in \f$ T \f$.
 *
 * Computes \f$ M(G,T) \f$ for given \f$ G \f$ and spanning forest \f$ T \f$ given by \p forestEdges.
 * The direction of the edges is specified by \p edgesReversed.
 * If \p forestEdges is \c NULL, \f$ T \f$ is some spanning forest \f$ T \f$ of \f$ G \f$ is computed.
 * The ordering of the columns can be specified via \p coforestEdges.
 *
 * \note The function computes a representation matrix of \f$ G \f$ regardless of whether \p forestEdges is a correct
 * spanning tree. This is indicated via *\p pisCorrectForest.
 */

CMR_EXPORT
CMR_ERROR CMRcomputeGraphTernaryRepresentationMatrix(
  CMR* cmr,                       /**< \ref CMR environment. */
  CMR_GRAPH* graph,              /**< \ref Graph \f$ G \f$. */
  CMR_CHRMAT** pmatrix,          /**< Pointer for storing \f$ M \f$ (may be \c NULL). */
  CMR_CHRMAT** ptranspose,       /**< Pointer for storing \f$ M^{\mathsf{T}} \f$ (may be \c NULL). */
  bool* edgesReversed,          /**< Indicates, for each edge \f$ \{u, v\}\f$, whether we consider \f$ (u, v)\f$  (if \c false) */
                                /**< or \f$ (v,u)\f$  (if \c true). */
  int numForestEdges,           /**< Length of \p forestEdges (0 if \c forestEdges is \c NULL). */
  CMR_GRAPH_EDGE* forestEdges,   /**< If not \c NULL, spanning forest edges as rows in this order. */
  int numCoforestEdges,         /**< Length of \p coforestEdges (0 if \c coforestEdges is \c NULL). */
  CMR_GRAPH_EDGE* coforestEdges, /**< If not \c NULL, complement of forest edges as columns in this order. */
  bool* pisCorrectForest        /**< Pointer for storing whether \c forestEdges is a spanning forest of \f$ G \f$ (may be \c NULL). */
);

/**
 * \brief Tests a binary matrix for graphicness.
 *
 * Let \f$ G = (V,E) \f$ be an undirected graph with nodes \f$ V \f$ and edges \f$ E \f$.
 * Let \f$ T \subseteq E \f$ be a spanning forest of \f$ G \f$.
 * The **binary representation matrix** \f$ M := M(G,T) \f$ is a matrix \f$ M \in \{0,1\}^{T \times (E \setminus T)} \f$
 * with \f$ M_{e,f} = 1 \iff e \f$ belongs to the unique cycle in \f$ T \cup \{f\} \f$.
 *
 * Tests if \f$ M = M(G,T) \f$ for some graph \f$ G \f$ and some spanning forest \f$ T \f$ of G and sets *\p pisGraphic
 * accordingly.
 * The matrix \f$ M \f$ is given by \f$ M^{\mathsf{T}} := \f$ \p transpose.
 *
 * If \f$ M \f$ is such a representation matrix and \p pgraph != \c NULL, then one possible graph \f$ G \f$ is
 * computed and stored in *\p pgraph.
 * The caller must release the memory via \ref CMRgraphFree.
 * If in addition to \p pgraph also \p pforestEdges (resp. \p pcoforestEdges) != \c NULL, then a corresponding
 * spanning forest \f$ T \f$ (resp.\ its complement \f$ E \setminus T \f$ is stored in *\p pforestEdges (resp.
 * \p pcoforestEdges).
 * The caller must release the memory via \ref CMRfreeBlockArray.
 *
 * If \f$ M \f$ is not such a representation matrix and \p psubmatrix != \c NULL, then a minimal submatrix of
 * \f$ M \f$ with the same property is computed and stored in *\p psubmatrix.
 * The caller must release the memory via \ref CMRsubmatFree.
 */

CMR_EXPORT
CMR_ERROR CMRtestBinaryGraphic(
  CMR* cmr,                         /**< \ref CMR environment. */
  CMR_CHRMAT* transpose,           /**< \f$ M^{\mathsf{T}} \f$ */
  bool* pisGraphic,               /**< Returns true if and only if the matrix is graphic. */
  CMR_GRAPH** pgraph,              /**< Pointer for storing \ref Graph \f$ G \f$ (if graphic). */
  CMR_GRAPH_EDGE** pforestEdges,   /**< Pointer for storing \f$ T \f$ (if graphic).  */
  CMR_GRAPH_EDGE** pcoforestEdges, /**< Pointer for storing \f$ E \setminus T \f$ (if graphic). */
  CMR_SUBMAT** psubmatrix          /**< Pointer for storing a minimal nongraphic submatrix (if nongraphic). */
);

/**
 * \brief Tests a ternary matrix for graphicness.
 *
 * Let \f$ G = (V,E) \f$ be a directed graph with nodes \f$ V \f$ and edges \f$ E \f$.
 * Let \f$ T \subseteq E \f$ be a spanning forest of \f$ G \f$, directed arbitrarily.
 * The **ternary representation matrix** \f$ M := M(G,T) \f$ is a matrix \f$ M \in \{-1,0,1\}^{T \times (E \setminus T)} \f$
 * with \f$ M_{e,f} = 1 \iff e \f$ belongs to the unique cycle in \f$ T \cup \{f\} \f$.
 *
 * Tests if \f$ M = M(G,T) \f$ for some graph \f$ G \f$ and some spanning forest \f$ T \f$ of G and sets *\p pisGraphic
 * accordingly.
 * The matrix \f$ M \f$ is given by \f$ M^{\mathsf{T}} := \f$ \p transpose.
 *
 * If \f$ M \f$ is such a representation matrix and \p pgraph != \c NULL, then one possible graph \f$ G \f$ is
 * computed and stored in *\p pgraph.
 * The caller must release the memory via \ref CMRgraphFree.
 * If in addition to \p pgraph also \p pforestEdges (resp. \p pcoforestEdges) != \c NULL, then a corresponding
 * spanning forest \f$ T \f$ (resp.\ its complement \f$ E \setminus T \f$ is stored in *\p pforestEdges (resp.
 * \p pcoforestEdges).
 * The caller must release the memory via \ref CMRfreeBlockArray.
 *
 * If \f$ M \f$ is not such a representation matrix and \p psubmatrix != \c NULL, then a minimal submatrix of
 * \f$ M \f$ with the same property is computed and stored in *\p psubmatrix.
 * The caller must release the memory via \ref CMRsubmatFree.
 */

CMR_EXPORT
CMR_ERROR CMRtestTernaryGraphic(
  CMR* cmr,                         /**< \ref CMR environment. */
  CMR_CHRMAT* transpose,           /**< \f$ M^{\mathsf{T}} \f$ */
  bool* pisGraphic,               /**< Returns true if and only if the matrix is graphic. */
  CMR_GRAPH** pgraph,              /**< Pointer for storing \ref Graph \f$ G \f$ (if graphic). */
  CMR_GRAPH_EDGE** pforestEdges,   /**< Pointer for storing \f$ T \f$ (if graphic).  */
  CMR_GRAPH_EDGE** pcoforestEdges, /**< Pointer for storing \f$ E \setminus T \f$ (if graphic). */
  bool** pedgesReversed,          /**< Pointer for storing indicators which edges are reversed for the correct sign. */
  CMR_SUBMAT** psubmatrix          /**< Pointer for storing a minimal nongraphic submatrix (if nongraphic). */
);

/**
 * \brief Finds an inclusion-wise maximal subset of columns that induce a graphic binary submatrix.
 *
 * Let \f$ G = (V,E) \f$ be an undirected graph with nodes \f$ V \f$ and edges \f$ E \f$.
 * Let \f$ T \subseteq E \f$ be a spanning forest of \f$ G \f$.
 * The **binary representation matrix** \f$ M := M(G,T) \f$ is a matrix \f$ M \in \{0,1\}^{T \times (E \setminus T)} \f$
 * with \f$ M_{e,f} = 1 \iff e \f$ belongs to the unique cycle in \f$ T \cup \{f\} \f$.
 *
 * Finds an inclusion-wise maximal subset \f$ J \f$ of columns of \f$ M \f$ such that \f$ M_{\star,J} \f$ is a binary
 * representation matrix.
 * To achieve this, it tries to append columns in the order given by \p orderedColumns, maintaining graphicness.
 */

CMR_EXPORT
CMR_ERROR CMRtestBinaryGraphicColumnSubmatrixGreedy(
  CMR* cmr,                 /**< \ref CMR environment. */
  CMR_CHRMAT* transpose,   /**< \f$ M^{\mathsf{T}} \f$ */
  size_t* orderedColumns, /**< Permutation of column indices of \f$ M \f$. */
  CMR_SUBMAT** psubmatrix  /**< Pointer for storing the submatrix. */
);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* CMR_GRAPHIC_H */
