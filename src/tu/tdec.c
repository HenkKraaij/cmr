#define TU_DEBUG_TDEC /* Uncomment to enable debugging of t-decompositions. */

#include <tu/tdec.h>
#include "env_internal.h"

#include <assert.h>
#include <limits.h>

typedef struct
{
  TU_TDEC_NODE representativeNode;  /**< \brief Next representative of same node towards root, or -1 if root. */
} TU_TDEC_NODE_DATA;

typedef struct
{
  int name;                   /**< \brief Name of this edge. */
  TU_TDEC_MEMBER member;      /**< \brief Member this edge belongs to or -1 if in free list. */
  TU_TDEC_NODE head;          /**< \brief Head node of this edge. */
  TU_TDEC_NODE tail;          /**< \brief Tail node of this edge. */
  TU_TDEC_EDGE prev;          /**< \brief Next edge of this member. Must be a directed cycle if member is a polygon. */
  TU_TDEC_EDGE next;          /**< \brief Previous edge of this member. Must be a directed cycle if member is a polygon. */
  TU_TDEC_MEMBER childMember; /**< \brief Child member linked to this edge, or -1. */
} TU_TDEC_EDGE_DATA;

typedef struct
{
  TU_TDEC_MEMBER_TYPE type;             /**< \brief Type of member. Only valid if root representative. */
  TU_TDEC_MEMBER representativeMember;  /**< \brief Next representative of same member towards root, or -1 if root. */
  TU_TDEC_MEMBER parentMember;          /**< \brief Parent member of this member. Only valid if root representative. */
  int numEdges;                         /**< \brief Number of edges. Only valid if root representative. */
  TU_TDEC_EDGE markerToParent;          /**< \brief Parent marker edge. Only valid if root representative. */
  TU_TDEC_EDGE markerOfParent;          /**< \brief Child marker of parent to which this member is linked. Only valid if root representative. */
  TU_TDEC_EDGE firstEdge;               /**< \brief First edge in doubly-linked edge list of this member. */
} TU_TDEC_MEMBER_DATA;

typedef struct
{
  TU_TDEC_EDGE edge;  /**< \brief Edge or -1. */
} TU_TDEC_ROW_DATA;

typedef struct
{
  TU_TDEC_EDGE edge;  /**< \brief Edge or -1. */
} TU_TDEC_COLUMN_DATA;

struct _TU_TDEC
{
  int memMembers;                   /**< \brief Allocated memory for members. */
  int numMembers;                   /**< \brief Number of members. */
  TU_TDEC_MEMBER_DATA* members;     /**< \brief Array of members. */
  int rootRow;                      /**< \brief Unique row element in member 0. */

  int memEdges;                     /**< \brief Allocated memory for edges. */
  int numEdges;                     /**< \brief Number of used edges. */
  TU_TDEC_EDGE_DATA* edges;         /**< \brief Array of edges. */
  TU_TDEC_EDGE firstFreeEdge;       /**< \brief First edge in free list or -1. */

  int memNodes;                     /**< \brief Allocated memory for nodes. */
  int numNodes;                     /**< \brief Number of nodes. */
  TU_TDEC_NODE_DATA* nodes;         /**< \brief Array of nodes. */
  TU_TDEC_NODE firstFreeNode;       /**< \brief First node in free list or -1. */

  int memRows;                      /**< \brief Allocated memory for \c rowEdges. */
  int numRows;                      /**< \brief Number of rows. */
  TU_TDEC_ROW_DATA* rowEdges;       /**< \brief Maps each row to its edge. */

  int memColumns;                   /**< \brief Allocated memory for \c columnEdges. */
  int numColumns;                   /**< \brief Number of columns. */
  TU_TDEC_COLUMN_DATA* columnEdges; /**< \brief Maps each column to its edge. */

  int numMarkers;                   /**< \brief Number of marker edge pairs in t-decomposition. */
};

static inline
bool isRepresentativeMember(
  TU_TDEC* tdec,        /**< t-decomposition. */
  TU_TDEC_MEMBER member /**< Member of \p tdec. */
)
{
  assert(tdec);
  return tdec->members[member].representativeMember < 0;
}

/**
 * \brief Checks whether \p tdec has consistent edge data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
const char* consistencyEdges(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (!isRepresentativeMember(tdec, member))
      continue;

    TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
    if (edge < 0)
      continue;

    do
    {
      if (edge < 0 || edge >= tdec->memEdges)
        return "edge out of range";
      if (tdec->edges[edge].next < 0 || tdec->edges[edge].next > tdec->memEdges)
        return "edge's next out of range";
      if (tdec->edges[tdec->edges[edge].next].prev != edge)
        return "inconsistent doubly-linked edge list";
      edge = tdec->edges[edge].next;
    }
    while (edge != tdec->members[member].firstEdge);
  }

  return NULL;
}

/**
 * \brief Checks whether \p tdec has consistent member data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
const char* consistencyMembers(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (tdec->members[member].type != TDEC_MEMBER_TYPE_BOND
      && tdec->members[member].type != TDEC_MEMBER_TYPE_PRIME
      && tdec->members[member].type != TDEC_MEMBER_TYPE_POLYGON)
      return "invalid member type";
  }

  return NULL;
}

/**
 * \brief Checks whether \p tdec has consistent node data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
const char* consistencyNodes(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    bool isPrime = tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME;
    TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
    if (edge < 0)
      continue;
    do
    {
      TU_TDEC_NODE head = tdec->edges[edge].head;
      TU_TDEC_NODE tail = tdec->edges[edge].tail;
      if (isPrime)
      {
        if (head < 0)
          return "invalid head node of prime member's edge";
        if (tail < 0)
          return "invalid tail node of prime member's edge";
        if (head >= tdec->memNodes)
          return "head node beyond range.";
        if (tail >= tdec->memNodes)
          return "tail node beyond range.";
      }
      else
      {
        if (head >= 0)
          return "head node of non-prime member's edge is not -1";
        if (tail >= 0)
          return "tail node of non-prime member's edge is not -1";
      }
      edge = tdec->edges[edge].next;
    }
    while (edge != tdec->members[member].firstEdge);
  }

  return NULL;
}


/**
 * \brief Checks whether \p tdec has consistent parent/child structure of members.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
const char* consistencyParentChild(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  if (tdec->memMembers < tdec->numMembers)
    return "member count and memory inconsistent";
  if (tdec->numMembers < 1)
    return "no members";
  if (tdec->members[0].markerOfParent != -1)
    return "marker of root's parent shall be invalid";
  if (tdec->members[0].parentMember != -1)
    return "root's parent shall be invalid";
  if (tdec->members[0].type != TDEC_MEMBER_TYPE_BOND)
    return "root member is not a bond";
  if (tdec->edges[tdec->members[0].markerToParent].name != INT_MIN)
    return "parent marker of root is not INT_MIN";

  int* countChildren = NULL;
  if (TUallocStackArray(tu, &countChildren, tdec->memMembers) != TU_OKAY)
    return "stack allocation in consistencyParentChild() failed";
  for (int m = 0; m < tdec->memMembers; ++m)
    countChildren[m] = 0;

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (tdec->members[member].parentMember < 0 && member != 0)
    {
      TUfreeStackArray(tu, &countChildren);
      return "non-root member without parent";
    }
    if (tdec->members[member].parentMember >= 0 && member == 0)
    {
      TUfreeStackArray(tu, &countChildren);
      return "root member with parent";
    }
    if (tdec->members[member].parentMember >= tdec->memMembers)
    {
      TUfreeStackArray(tu, &countChildren);
      return "parent member out of range";
    }
    if (tdec->members[member].parentMember >= 0)
      countChildren[tdec->members[member].parentMember]++;
  }

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
    if (edge < 0)
      continue;
    do
    {
      if (tdec->edges[edge].childMember >= 0)
      {
        countChildren[member]--;
        
        if (tdec->members[tdec->edges[edge].childMember].parentMember != member)
        {
          TUfreeStackArray(tu, &countChildren);
          return "bad parentMember";
        }
        if (tdec->members[tdec->edges[edge].childMember].markerOfParent != edge)
        {
          TUfreeStackArray(tu, &countChildren);
          return "bad markerOfParent";
        }
        TU_TDEC_EDGE markerChild = tdec->members[tdec->edges[edge].childMember].markerToParent;
        if (tdec->edges[markerChild].name != -tdec->edges[edge].name)
        {
          TUfreeStackArray(tu, &countChildren);
          return "names of corresponding marker edges are not negations";
        }
      }
      edge = tdec->edges[edge].next;
    }
    while (edge != tdec->members[member].firstEdge);
  }

  if (TUfreeStackArray(tu, &countChildren) != TU_OKAY)
    return "stack deallocation in consistencyParentChild() failed";

  return NULL;
}


/**
 * \brief Checks whether the members of \p tdec form a tree.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
const char* consistencyTree(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (member == 0)
      continue;

    int length = 0;
    TU_TDEC_MEMBER current;
    for (current = tdec->members[member].parentMember; current >= 0;
      current = tdec->members[current].parentMember)
    {
      ++length;
      if (length > tdec->numMembers)
        return "infinite member parent loop";
    }
  }

  return NULL;
}

const char* TUtdecConsistency(TU* tu, TU_TDEC* tdec)
{
  const char* message = NULL;
  if ((message = consistencyMembers(tu, tdec)))
    return message;
  if ((message = consistencyEdges(tu, tdec)))
    return message;
  if ((message = consistencyNodes(tu, tdec)))
    return message;
  if ((message = consistencyParentChild(tu, tdec)))
    return message;
  if ((message = consistencyTree(tu, tdec)))
    return message;

  return NULL;
}

#define assertConsistency(tu, tdec) \
  do \
  { \
    const char* __message = TUtdecConsistency(tu, tdec); \
    if (__message) \
    { \
      fflush(stdout); \
      fprintf(stderr, "%s:%d: t-decomposition is inconsistent: %s\n", __FILE__, __LINE__, __message); \
      fflush(stderr); \
      exit(1); \
    } \
  } \
  while (false);


typedef enum
{
  TYPE_UNKNOWN = 0,
  TYPE_1_HEAD_END_TAIL_END = 1, /**< Edge plus path is a cycle. */
  TYPE_2_HEAD_END_TAIL_IN = 2,  /**< Head is path end and tail is inner node. */
  TYPE_3_HEAD_END_TAIL_OUT = 3, /**< Head is path end and tail does not belong to path. */
  TYPE_4_HEAD_IN_TAIL_IN = 4,   /**< Head and tail are inner nodes such that adding the edge yields a path. */
  TYPE_ROOT = 5,                /**< Root member. */
  TYPE_5_OTHER = 6              /**< All other cases. */
} Type;

/**
 * \brief Additional edge information specific to a path.
 */

typedef struct _ReducedEdge
{
  TU_TDEC_EDGE edge;          /**< \brief The edge in the t-decomposition. */
  struct _ReducedEdge* next;  /**< \brief Next edge of this reduced member, or \c NULL. */
} ReducedEdge;

/**
 * \brief Additional member information specfic to a given path.
 */

typedef struct _ReducedMember
{
  TU_TDEC_MEMBER member;                /**< \brief The member from the t-decomposition. */
  int depth;                            /**< \brief Depth of this member in the reduced t-decomposition. */
  Type type;                            /**< \brief Type of this member. */
  int numChildren;                      /**< \brief Number of children in the reduced t-decomposition. */
  struct _ReducedMember** children;     /**< \brief Children in the reduced t-decomposition. */
  ReducedEdge* firstReducedEdge;        /**< \brief First edge in linked list of edges of this reduced member. */
} ReducedMember;

struct _TU_TDEC_NEWCOLUMN
{
  bool remainsGraphic;                      /**< \brief Indicator whether adding this column maintains graphicness. */
  int memReducedMembers;                    /**< \brief Allocated memory for \c reducedMembers. */
  int numReducedMembers;                    /**< \brief Number of members in \c reducedMembers. */
  ReducedMember* reducedMembers;            /**< \brief Array of reduced members, sorted by increasing depth. */
  ReducedMember** membersToReducedMembers;  /**< \brief Array mapping members to members of the reduced t-decomposition. */

  ReducedEdge* reducedEdgeStorage;          /**< \brief Storage for edge lists of reduced members. */
  int memReducedEdgeStorage;                /**< \brief Allocated memory for \c reducedEdgeStorage. */
  int usedReducedEdgeStorage;               /**< \brief Number of stored edges in \c reducedEdgeStorage. */

  ReducedMember** childrenStorage;          /**< \brief Storage for members' arrays of children in reduced t-decomposition. */
  int usedChildrenStorage;                  /**< \brief Number of stored children in \c childrenStorage. */
  int memChildrenStorage;                   /**< \brief Allocated memory for \c childrenStorage. */

  int* nodesDegree;                         /**< \brief Map from nodes to degree w.r.t. path edges. */
  bool* edgesInPath;                        /**< \brief Map from edges to indicator for being in the path. */

  TU_TDEC_NODE terminalNode1;               /**< \brief First terminal node of path. */
  TU_TDEC_NODE terminalNode2;               /**< \brief Second terminal node of path. */
  TU_TDEC_MEMBER terminalMember1;           /**< \brief First terminal member of path. */
  TU_TDEC_MEMBER terminalMember2;           /**< \brief Second terminal member of path. */
};

int compareMemberDepths(const void* a, const void* b)
{
  const ReducedMember* first = a;
  const ReducedMember* second = b;
  /* Negative depths are moved to the end. */
  if (first->depth <= 0)
    return +1;
  if (second->depth <= 0)
    return -1;
  return first->depth - second->depth;
}

static TU_TDEC_MEMBER findMember(TU_TDEC* tdec, TU_TDEC_MEMBER start)
{
  TU_TDEC_MEMBER current = start;
  TU_TDEC_MEMBER next;
  while ((next = tdec->members[current].representativeMember) >= 0)
    current = next;
  TU_TDEC_MEMBER root = current;
  current = start;
  while ((next = tdec->members[current].representativeMember) >= 0)
  {
    if (next != root)
      tdec->members[current].representativeMember = root;
    current = next;
  }
  return root;
}

static inline
TU_TDEC_MEMBER findMemberParent(TU_TDEC* tdec, TU_TDEC_MEMBER member)
{
  TU_TDEC_MEMBER someParent = tdec->members[member].parentMember;
  if (someParent >= 0)
    return findMember(tdec, someParent);
  else
    return -1;
}

static // TODO: inline
TU_TDEC_MEMBER findEdgeMember(TU_TDEC* tdec, TU_TDEC_EDGE edge)
{
  return findMember(tdec, tdec->edges[edge].member);
}

static
TU_TDEC_NODE findNode(TU_TDEC* tdec, TU_TDEC_NODE start)
{
  TU_TDEC_NODE current = start;
  TU_TDEC_NODE next;
  while ((next = tdec->nodes[current].representativeNode) >= 0)
    current = next;
  TU_TDEC_NODE root = current;
  current = start;
  while ((next = tdec->nodes[current].representativeNode) >= 0)
  {
    if (next != root)
      tdec->nodes[current].representativeNode = root;
    current = next;
  }
  return root;
}

static // TODO: inline
TU_TDEC_NODE findEdgeHead(TU_TDEC* tdec, TU_TDEC_EDGE edge)
{
  assert(edge >= 0);
  assert(edge < tdec->memEdges);
  assert(tdec->edges[edge].head >= 0);
  assert(tdec->edges[edge].head < tdec->memNodes);
  return findNode(tdec, tdec->edges[edge].head);
}

static // TODO: inline
TU_TDEC_NODE findEdgeTail(TU_TDEC* tdec, TU_TDEC_EDGE edge)
{
  return findNode(tdec, tdec->edges[edge].tail);
}

static
TU_ERROR createNode(
  TU* tu,             /**< \ref TU environment . */
  TU_TDEC* tdec,      /**< t-decomposition. */
  TU_TDEC_NODE* pnode /**< Pointer for storing new node. */
)
{
  assert(tu);
  assert(tdec);
  assert(pnode);

  TU_TDEC_NODE node = tdec->firstFreeNode;
  if (node >= 0)
  {
#if defined(TU_DEBUG_TDEC)
    printf("        createNode returns free node %d.\n", node);
#endif /* TU_DEBUG_TDEC */
    tdec->firstFreeNode = tdec->nodes[node].representativeNode;
  }
  else /* No member in free list, so we enlarge the array. */
  {
    int newSize = 2 * tdec->memNodes + 16;
    TU_CALL( TUreallocBlockArray(tu, &tdec->nodes, newSize) );
    for (int v = tdec->memNodes + 1; v < newSize; ++v)
      tdec->nodes[v].representativeNode = v+1;
    tdec->nodes[newSize-1].representativeNode = -1;
    tdec->firstFreeNode = tdec->memNodes + 1;
    node = tdec->memNodes;
    tdec->memNodes = newSize;
#if defined(TU_DEBUG_TDEC)
    printf("        createNode enlarges node array to %d and returns node %d.\n", newSize, node);
#endif /* TU_DEBUG_TDEC */
  }
  tdec->nodes[node].representativeNode = -1;
  tdec->numNodes++;

  *pnode = node;

  return TU_OKAY;
}

static void setRowEdge(
  TU* tu,           /**< \ref TU environment. */
  TU_TDEC* tdec,    /**< t-decomposition. */
  int row ,         /**< Row (index). */
  TU_TDEC_EDGE edge /**< Edge to be assigned to \p row. */
)
{
  assert(tu);
  assert(tdec);
  assert(row >= 0);
  assert(edge >= 0);

  if (row >= tdec->memRows)
  {
    int newSize = 2*tdec->memRows + 16;
    TUreallocBlockArray(tu, &tdec->rowEdges, newSize);
    for (int c = tdec->memRows; c < newSize; ++c)
      tdec->rowEdges[c].edge = -1;
    tdec->memRows = newSize;
  }

  assert(tdec->rowEdges[row].edge == -1);
  tdec->rowEdges[row].edge = edge;
  if (row >= tdec->numRows)
    tdec->numRows = row + 1;
}

static
TU_ERROR setColumnEdge(
  TU* tu,           /**< \ref TU environment. */
  TU_TDEC* tdec,    /**< t-decomposition. */
  int column,       /**< Column (index). */
  TU_TDEC_EDGE edge /**< Edge to be assigned to \p column. */
)
{
  assert(tu);
  assert(tdec);
  assert(column >= 0);
  assert(edge >= 0);

  if (column >= tdec->memColumns)
  {
    int newSize = 2*tdec->memColumns + 16;
    TU_CALL( TUreallocBlockArray(tu, &tdec->columnEdges, newSize) );
    for (int c = tdec->memColumns; c < newSize; ++c)
      tdec->columnEdges[c].edge = -1;
    tdec->memColumns = newSize;
  }

  assert(tdec->columnEdges[column].edge == -1);
  tdec->columnEdges[column].edge = edge;
  if (column >= tdec->numColumns)
    tdec->numColumns = column + 1;

  return TU_OKAY;
}

/**
 * \brief Adds \p edge to the edge list of \p member.
 */

static
TU_ERROR addEdgeToMembersEdgeList(
  TU* tu,               /**< \ref TU environment. */
  TU_TDEC* tdec,        /**< t-decomposition. */
  TU_TDEC_EDGE edge,    /**< Edge to be added. */
  TU_TDEC_MEMBER member /**< Member. */
)
{
  assert(tu);
  assert(tdec);
  assert(edge >= 0);
  assert(member >= 0);
  assert(member < tdec->numMembers);
  assert(isRepresentativeMember(tdec, member));

  TU_TDEC_EDGE first = tdec->members[member].firstEdge;
  if (first >= 0)
  {
    assert(tdec->members[member].numEdges > 0);
    TU_TDEC_EDGE last = tdec->edges[first].prev;
    tdec->edges[edge].next = first;
    tdec->edges[edge].prev = last;
    tdec->edges[first].prev = edge;
    tdec->edges[last].next =  edge;
  }
  else
  {
    assert(tdec->members[member].numEdges == 0);
    tdec->edges[edge].next = edge;
    tdec->edges[edge].prev = edge;
  }
  tdec->members[member].firstEdge = edge;
  tdec->members[member].numEdges++;

  return TU_OKAY;
}

/**
 * \brief Creates a new edge.
 */

static
TU_ERROR createEdge(
  TU* tu,                 /**< \ref TU environment. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_MEMBER member,  /**< Member this edge belongs to. */
  TU_TDEC_EDGE* pedge     /**< Pointer for storing the new edge. */
)
{
  assert(tu);
  assert(tdec);
  assert(pedge);
  assert(member < 0 || isRepresentativeMember(tdec, member));

  TU_TDEC_EDGE edge = tdec->firstFreeEdge;
  if (edge >= 0)
  {
#if defined(TU_DEBUG_TDEC)
    printf("        createEdge returns free edge %d.\n", edge);
#endif /* TU_DEBUG_TDEC */
    tdec->firstFreeEdge = tdec->edges[edge].next;
  }
  else /* No edge in free list, so we enlarge the array. */
  {
    int newSize = 2 * tdec->memEdges + 16;
    TU_CALL( TUreallocBlockArray(tu, &tdec->edges, newSize) );
    for (int e = tdec->memEdges + 1; e < newSize; ++e)
    {
      tdec->edges[e].next = e+1;
      tdec->edges[e].member = -1;
    }
    tdec->edges[newSize-1].next = -1;
    tdec->firstFreeEdge = tdec->memEdges + 1;
    edge = tdec->memEdges;
    tdec->memEdges = newSize;
#if defined(TU_DEBUG_TDEC)
    printf("        createEdge enlarges edge array to %d and returns edge %d.\n", newSize, edge);
#endif /* TU_DEBUG_TDEC */
  }

  tdec->edges[edge].member = member;
  tdec->numEdges++;

  *pedge = edge;

  return TU_OKAY;
}

static
TU_ERROR createRowEdge(
  TU* tu,                 /**< \ref TU environment. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_EDGE* pedge,    /**< Pointer for storing the new edge. */
  TU_TDEC_MEMBER member,  /**< Member this edge belongs to. */
  TU_TDEC_NODE head,      /**< Head node of this edge. */
  TU_TDEC_NODE tail,      /**< Tail node of this edge. */
  int row                 /**< Row (index) this edge corresponds to. */
)
{
  assert(tu);
  assert(tdec);
  assert(pedge);

  TU_CALL( createEdge(tu, tdec, member, pedge) );
  TU_TDEC_EDGE edge = *pedge;
  TU_TDEC_EDGE_DATA* data = &tdec->edges[edge];
  data->head = head;
  data->tail = tail;
  data->childMember = -1;
  data->name = row;
  setRowEdge(tu, tdec, row, edge);

#if defined(TU_DEBUG_TDEC)
  printf("        Created row edge {%d,%d} of member %d for row %d.\n", head, tail, member, row);
#endif /* TU_DEBUG_TDEC */

  return TU_OKAY;
}

static
TU_ERROR createColumnEdge(
  TU* tu,                 /*< TU environment. */
  TU_TDEC* tdec,          /*< t-decomposition. */
  TU_TDEC_EDGE* pedge,    /**< Pointer for storing the new edge. */
  TU_TDEC_MEMBER member,  /*< Member this edge belongs to. */
  TU_TDEC_NODE head,      /*< Head node of this edge. */
  TU_TDEC_NODE tail,      /*< Tail node of this edge. */
  int column              /*< Column (index) this edge corresponds to. */
)
{
  assert(tu);
  assert(tdec);
  assert(pedge);

  TU_CALL( createEdge(tu, tdec, member, pedge) );
  TU_TDEC_EDGE edge = *pedge;
  TU_TDEC_EDGE_DATA* data = &tdec->edges[edge];
  data->head = head;
  data->tail = tail;
  data->childMember = -1;
  data->name = -1-column;
  TU_CALL( setColumnEdge(tu, tdec, column, edge) );

#if defined(TU_DEBUG_TDEC)
  printf("        Created column edge {%d,%d} of member %d for column %d.\n", head, tail, member,
    column);
#endif /* TU_DEBUG_TDEC */

  return TU_OKAY;
}

static
TU_ERROR createMarkerEdge(
  TU* tu,                 /**< \ref TU environment. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_EDGE* pedge,    /**< Pointer for storing the new edge. */
  TU_TDEC_MEMBER member,  /**< Member this edge belongs to. */
  TU_TDEC_NODE head,      /**< Head node of this edge. */
  TU_TDEC_NODE tail,      /**< Tail node of this edge. */
  bool isParent           /**< Whether this is the parent marker edge. */
)
{
  assert(tu);
  assert(tdec);
  assert(pedge);

  TU_CALL( createEdge(tu, tdec, member, pedge) );
  TU_TDEC_EDGE edge = *pedge;
  TU_TDEC_EDGE_DATA* data = &tdec->edges[edge];
  data->head = head;
  data->tail = tail;
  data->childMember = -1;
  if (isParent)
    data->name = INT_MAX - tdec->numMarkers;
  else
    data->name = -(INT_MAX - tdec->numMarkers);

#if defined(TU_DEBUG_TDEC)
  printf("        Created %s marker edge {%d,%d} of member %d.\n", isParent ? "parent" : "child",
    head, tail, member);
#endif /* TU_DEBUG_TDEC */

  return TU_OKAY;
}

static
TU_ERROR createMember(
  TU* tu,                   /**< \ref TU environment . */
  TU_TDEC* tdec,            /**< t-decomposition. */
  TU_TDEC_MEMBER_TYPE type, /**< Type of member. */
  TU_TDEC_MEMBER* pmember   /**< Created member. */
)
{
  assert(tu);
  assert(tdec);

  if (tdec->numMembers == tdec->memMembers)
  {
    tdec->memMembers = 16 + 2 * tdec->memMembers;
    TU_CALL( TUreallocBlockArray(tu, &tdec->members, tdec->memMembers) );
  }

  TU_TDEC_MEMBER_DATA* data = &tdec->members[tdec->numMembers];
  data->markerOfParent = INT_MIN;
  data->markerToParent = INT_MIN;
  data->firstEdge = -1;
  data->representativeMember = -1;
  data->numEdges = 0;
  data->parentMember = INT_MIN;
  data->type = type;
  *pmember = tdec->numMembers;
  tdec->numMembers++;

  return TU_OKAY;
}

TU_ERROR TUtdecCreate(TU* tu, TU_TDEC** ptdec, int rootRow, int memEdges, int memNodes,
  int memMembers, int numRows, int numColumns)
{
  assert(tu);
  assert(ptdec);
  assert(!*ptdec);

  TU_CALL( TUallocBlock(tu, ptdec) );
  TU_TDEC* tdec = *ptdec;
  if (memMembers < 1)
    memMembers = 1;
  tdec->memMembers = memMembers;
  tdec->numMembers = 1;
  tdec->members = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->members, tdec->memMembers) );
  tdec->members[0].representativeMember = -1;
  tdec->members[0].parentMember = -1;
  tdec->members[0].numEdges = 2;
  tdec->members[0].type = TDEC_MEMBER_TYPE_BOND;
  tdec->members[0].markerToParent = 0;
  tdec->members[0].markerOfParent = -1;
  tdec->members[0].firstEdge = 0;
  tdec->rootRow = rootRow;

  if (memNodes < 1)
    memNodes = 1;
  tdec->memNodes = memNodes;
  tdec->nodes = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->nodes, memNodes) );
  tdec->numNodes = 0;
  for (int v = 0; v < memNodes; ++v)
    tdec->nodes[v].representativeNode = v+1;
  tdec->nodes[memNodes-1].representativeNode = -1;
  tdec->firstFreeNode = 0;

  if (memEdges < 2)
    memEdges = 2;
  tdec->memEdges = memEdges;
  tdec->edges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->edges, memEdges) );
  tdec->numEdges = 2;

  /* First edge is co-tree edge corresponding to artificial column. */
  tdec->edges[0].name = INT_MIN;
  tdec->numMarkers = 0;
  tdec->edges[0].member = 0;
  tdec->edges[0].head = -1;
  tdec->edges[0].tail = -1;
  tdec->edges[0].childMember = -1;
  tdec->edges[0].prev = 1;
  tdec->edges[0].next = 1;

  /* Second edge is tree edge corresponding to \c rootRow. */
  tdec->edges[1].name = rootRow;
  tdec->edges[1].member = 0;
  tdec->edges[1].head = -1;
  tdec->edges[1].tail = -1;
  tdec->edges[1].childMember = -1;
  tdec->edges[1].prev = 0;
  tdec->edges[1].next = 0;
  if (memEdges > 2)
  {
    for (int e = 2; e < memEdges; ++e)
    {
      tdec->edges[e].next = e+1;
      tdec->edges[e].member = -1;
    }
    tdec->edges[memEdges-1].next = -1;
    tdec->firstFreeEdge = 2;
  }
  else
    tdec->firstFreeEdge = -1;

  tdec->numRows = numRows > rootRow ? numRows : rootRow + 1;
  tdec->memRows = numRows;
  tdec->rowEdges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->rowEdges, tdec->numRows) );
  for (int r = 0; r < tdec->numRows; ++r)
    tdec->rowEdges[r].edge = -1;
  tdec->rowEdges[rootRow].edge = 1;

  tdec->numColumns = numColumns > 0 ? numColumns : 1;
  tdec->memColumns = tdec->numColumns;
  tdec->columnEdges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->columnEdges, tdec->numColumns) );
  for (int c = 0; c < tdec->numColumns; ++c)
    tdec->columnEdges[c].edge = -1;

  assertConsistency(tu, tdec);

  return TU_OKAY;
}

TU_ERROR TUtdecFree(TU* tu, TU_TDEC** ptdec)
{
  assert(ptdec);
  assert(*ptdec);

  TU_TDEC* tdec = *ptdec;
  TU_CALL( TUfreeBlockArray(tu, &tdec->members) );
  TU_CALL( TUfreeBlockArray(tu, &tdec->edges) );
  TU_CALL( TUfreeBlockArray(tu, &tdec->nodes) );
  TU_CALL( TUfreeBlockArray(tu, &tdec->rowEdges) );
  TU_CALL( TUfreeBlockArray(tu, &tdec->columnEdges) );
  TU_CALL( TUfreeBlock(tu, ptdec) );

  return TU_OKAY;
}

int TUtdecBasisSize(TU_TDEC* tdec)
{
  assert(tdec);

  return tdec->numRows;
}

int TUtdecCobasisSize(TU_TDEC* tdec)
{
  assert(tdec);

  return tdec->numColumns;
}

int TUtdecNumEdges(TU_TDEC* tdec)
{
  assert(tdec);

  return tdec->numEdges;
}

TU_ERROR TUtdecToGraph(TU* tu, TU_TDEC* tdec, TU_GRAPH* graph, bool merge, TU_GRAPH_EDGE* basis,
  TU_GRAPH_EDGE* cobasis, int* edgeElements)
{
  assert(tu);
  assert(tdec);
  assert(graph);

  assertConsistency(tu, tdec);

#if defined(TU_DEBUG_TDEC)
  printf("TUtdecToGraph for t-decomposition.\n");
#endif /* TU_DEBUG_TDEC */

  TU_CALL( TUgraphClear(tu, graph) );

  TU_GRAPH_EDGE* localEdgeElements = NULL;
  if (edgeElements)
    localEdgeElements = edgeElements;
  else if (basis || cobasis)
    TU_CALL( TUallocStackArray(tu, &localEdgeElements, tdec->memEdges) );
  TU_GRAPH_NODE* tdecNodesToGraphNodes = NULL;
  TU_CALL( TUallocStackArray(tu, &tdecNodesToGraphNodes, tdec->numNodes) );
  TU_GRAPH_EDGE* tdecEdgesToGraphEdges = NULL;
  TU_CALL( TUallocStackArray(tu, &tdecEdgesToGraphEdges, tdec->memEdges) );

  for (int v = 0; v < tdec->memNodes; ++v)
  {
    if (tdec->nodes[v].representativeNode < 0)
    {
      TU_CALL( TUgraphAddNode(tu, graph, &tdecNodesToGraphNodes[v]) );
    }
    else
      tdecNodesToGraphNodes[v] = -1;
  }

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (!isRepresentativeMember(tdec, member))
      continue;

    TU_TDEC_MEMBER_TYPE type = tdec->members[member].type;
#if defined(TU_DEBUG_TDEC)
    printf("  Member %d is %s with %d edges.\n", member, type == TDEC_MEMBER_TYPE_BOND ?
      "a bond" : (type == TDEC_MEMBER_TYPE_POLYGON ? "a polygon" : "prime"),
       tdec->members[member].numEdges);
#endif /* TU_DEBUG_TDEC */
    
    TU_GRAPH_EDGE graphEdge;
    TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
    if (type == TDEC_MEMBER_TYPE_PRIME)
    {
      do
      {
        TU_TDEC_NODE head = findEdgeHead(tdec, edge);
        TU_TDEC_NODE tail = findEdgeTail(tdec, edge);
        TU_CALL( TUgraphAddEdge(tu, graph, tdecNodesToGraphNodes[head], tdecNodesToGraphNodes[tail],
          &graphEdge) );
        tdecEdgesToGraphEdges[edge] = graphEdge;
        if (localEdgeElements)
          localEdgeElements[graphEdge] = tdec->edges[edge].name;
        edge = tdec->edges[edge].next;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    else if (type == TDEC_MEMBER_TYPE_BOND)
    {
      TU_GRAPH_NODE graphHead, graphTail;
      TU_CALL( TUgraphAddNode(tu, graph, &graphHead) );
      TU_CALL( TUgraphAddNode(tu, graph, &graphTail) );
      do
      {
        if (tdec->edges[edge].name != INT_MIN)
        {
          TU_CALL( TUgraphAddEdge(tu, graph, graphHead, graphTail, &graphEdge) );
          tdecEdgesToGraphEdges[edge] = graphEdge;
          if (localEdgeElements)
            localEdgeElements[graphEdge] = tdec->edges[edge].name;
        }
        else
          tdecEdgesToGraphEdges[edge] = -1;
        edge = tdec->edges[edge].next;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    else
    {
      assert(type == TDEC_MEMBER_TYPE_POLYGON);

      TU_GRAPH_NODE firstNode, v;
      TU_CALL( TUgraphAddNode(tu, graph, &firstNode) );
      v = firstNode;
      edge = tdec->edges[edge].next;
      while (edge != tdec->members[member].firstEdge)
      {
        
        TU_GRAPH_NODE w;
        TU_CALL( TUgraphAddNode(tu, graph, &w) );
        TU_CALL( TUgraphAddEdge(tu, graph, v, w, &graphEdge) );
        tdecEdgesToGraphEdges[edge] = graphEdge;
        if (localEdgeElements)
          localEdgeElements[graphEdge] = tdec->edges[edge].name;

        edge = tdec->edges[edge].next;
        v = w;
      }
      TU_CALL( TUgraphAddEdge(tu, graph, v, firstNode, &graphEdge) );
      tdecEdgesToGraphEdges[edge] = graphEdge;
      if (localEdgeElements)
        localEdgeElements[graphEdge] = tdec->edges[edge].name;
    }
  }

  /* Merge respective parent and child edges. */

  if (merge)
  {
#if defined(TU_DEBUG_TDEC)
    printf("  Before merging, the graph has %d nodes and %d edges.\n", TUgraphNumNodes(graph),
      TUgraphNumEdges(graph));
    fflush(stdout);
#endif /* TU_DEBUG_TDEC */

    for (int m = 1; m < tdec->numMembers; ++m)
    {
      if (tdec->members[m].type == TDEC_MEMBER_TYPE_INVALID)
        continue;

      TU_GRAPH_EDGE parent = tdecEdgesToGraphEdges[tdec->members[m].markerOfParent];
      TU_GRAPH_EDGE child = tdecEdgesToGraphEdges[tdec->members[m].markerToParent];
      TU_GRAPH_NODE parentU = TUgraphEdgeU(graph, parent);
      TU_GRAPH_NODE parentV = TUgraphEdgeV(graph, parent);
      TU_GRAPH_NODE childU = TUgraphEdgeU(graph, child);
      TU_GRAPH_NODE childV = TUgraphEdgeV(graph, child);

#if defined(TU_DEBUG_TDEC)
      printf("  Merging edges %d = {%d,%d} <%d> and %d = {%d,%d} <%d>.\n", parent, parentU, parentV,
        tdec->edges[tdec->members[m].markerOfParent].name, child, childU, childV,
        tdec->edges[tdec->members[m].markerToParent].name);
      fflush(stdout);
#endif /* TU_DEBUG_TDEC */

      TU_CALL( TUgraphMergeNodes(tu, graph, parentU, childU) );
      TU_CALL( TUgraphDeleteNode(tu, graph, childU) );
      TU_CALL( TUgraphMergeNodes(tu, graph, parentV, childV) );
      TU_CALL( TUgraphDeleteNode(tu, graph, childV) );

      TU_CALL( TUgraphDeleteEdge(tu, graph, parent) );
      TU_CALL( TUgraphDeleteEdge(tu, graph, child) );
    }
  }

  // TODO: Remove nodes with degree 0 or 1?!

  /* Construct (co)basis. */

  if (basis || cobasis)
  {
    for (TU_GRAPH_ITER i = TUgraphEdgesFirst(graph); TUgraphEdgesValid(graph, i);
      i = TUgraphEdgesNext(graph, i))
    {
      TU_GRAPH_EDGE e = TUgraphEdgesEdge(graph, i);

#if defined(TU_DEBUG_TDEC)
      printf("  Graph edge %d = {%d,%d}\n", e, TUgraphEdgeU(graph, e), TUgraphEdgeV(graph, e));
      fflush(stdout);
#endif /* TU_DEBUG_TDEC */

      int element = localEdgeElements[e];
      if (element >= 0 && basis)
        basis[element] = e;
      else if (element < 0 && cobasis)
        cobasis[-1-element] = e;
    }
  }

  TU_CALL( TUfreeStackArray(tu, &tdecEdgesToGraphEdges) );
  TU_CALL( TUfreeStackArray(tu, &tdecNodesToGraphNodes) );
  if (localEdgeElements != edgeElements)
    TU_CALL( TUfreeStackArray(tu, &localEdgeElements) );

  return TU_OKAY;
}

TU_ERROR TUtdecnewcolumnCreate(TU* tu, TU_TDEC_NEWCOLUMN** pnewcolumn)
{
  assert(tu);

  TU_CALL( TUallocBlock(tu, pnewcolumn) );
  TU_TDEC_NEWCOLUMN* newcolumn = *pnewcolumn;
  newcolumn->remainsGraphic = true;
  newcolumn->memReducedMembers = 0;
  newcolumn->numReducedMembers = 0;
  newcolumn->reducedMembers = NULL;
  newcolumn->membersToReducedMembers = NULL;

  newcolumn->reducedEdgeStorage = NULL;
  newcolumn->memReducedEdgeStorage = 0;
  newcolumn->usedReducedEdgeStorage = 0;

  newcolumn->memChildrenStorage = 0;
  newcolumn->usedChildrenStorage = 0;
  newcolumn->childrenStorage = NULL;

  newcolumn->nodesDegree = NULL;
  newcolumn->edgesInPath = NULL;

  return TU_OKAY;
}

TU_ERROR TUtdecnewcolumnFree(TU* tu, TU_TDEC_NEWCOLUMN** pnewcolumn)
{
  assert(tu);
  assert(*pnewcolumn);

  TU_TDEC_NEWCOLUMN* newcolumn = *pnewcolumn;
  
  if (newcolumn->edgesInPath)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->edgesInPath) );
  }

  if (newcolumn->nodesDegree)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->nodesDegree) );
  }

  if (newcolumn->reducedMembers)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->reducedMembers) );
  }
  if (newcolumn->membersToReducedMembers)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->membersToReducedMembers) );
  }
  if (newcolumn->reducedEdgeStorage)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->reducedEdgeStorage) );
  }
  if (newcolumn->childrenStorage)
  {
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->childrenStorage) );
  }

  TU_CALL( TUfreeBlock(tu, pnewcolumn) );

  return TU_OKAY;
}

static
TU_ERROR initializeNewColumn(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn  /**< new column. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);

#ifndef NDEBUG
  newcolumn->terminalNode1 = INT_MIN;
  newcolumn->terminalNode2 = INT_MIN;
  newcolumn->terminalMember1 = -1;
  newcolumn->terminalMember2 = -1;
  newcolumn->numReducedMembers = 0;
#endif /* !NDEBUG */

  newcolumn->remainsGraphic = true;
  newcolumn->usedReducedEdgeStorage = 0;

  // TODO: Remember sizes of these arrays.
  TU_CALL( TUreallocBlockArray(tu, &newcolumn->edgesInPath, tdec->memEdges) );
  TU_CALL( TUreallocBlockArray(tu, &newcolumn->nodesDegree, tdec->memNodes) );

  return TU_OKAY;
}

static
TU_ERROR findReducedDecomposition(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  int* entryRows,               /**< Array of rows of new column's enries. */
  int numEntries                /**< Length of \p entryRows. */
)
{
  /* Enlarge members array. */
  if (newcolumn->memReducedMembers < tdec->numMembers)
  {
    newcolumn->memReducedMembers = tdec->memMembers;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedMembers, newcolumn->memReducedMembers) );
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->membersToReducedMembers,
      newcolumn->memReducedMembers) );
  }

  /* Identify all members on the path. For the induced sub-arborescence we also compute the
   * depths. After the computation, its root has depth pathRootDepth. */
#if defined(TU_DEBUG_TDEC)
  printf("    Finding reduced t-decomposition.\n");
#endif /* TU_DEBUG_TDEC */

  int* memberDepths = NULL;
  TU_CALL( TUallocStackArray(tu, &memberDepths, tdec->numMembers) );
  for (int m = 0; m < tdec->numMembers; ++m)
    memberDepths[m] = 0;
  TU_TDEC_MEMBER reducedRootMember = -1;
  int reducedRootDepth = 0;
  newcolumn->numReducedMembers = 0;
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    TU_TDEC_EDGE edge = (row < tdec->numRows) ? tdec->rowEdges[row].edge : -1;
    if (edge >= 0)
    {
#if defined(TU_DEBUG_TDEC)
      printf("      Edge %d exists.\n", edge);
#endif /* TU_DEBUG_TDEC */
      TU_TDEC_MEMBER member = findEdgeMember(tdec, edge);
      if (!reducedRootDepth)
      {
        /* The first member receives a #members, its parent #members-1, etc. */
        reducedRootMember = member;
        int depth = tdec->numMembers;
        reducedRootDepth = depth;
        while (member >= 0)
        {
#if defined(TU_DEBUG_TDEC)
          printf("        Member %d receives depth %d on initial path.\n", member, depth);
          fflush(stdout);
#endif /* TU_DEBUG_TDEC */
          assert(memberDepths[member] == 0);
          memberDepths[member] = depth;
          newcolumn->reducedMembers[newcolumn->numReducedMembers].member = member;
          newcolumn->reducedMembers[newcolumn->numReducedMembers].depth = depth;
          newcolumn->numReducedMembers++;
          --depth;
          member = findMemberParent(tdec, member);
        }
      }
      else
      {
        int count = 0;
        TU_TDEC_MEMBER m = member;
        while (memberDepths[m] == 0)
        {
          memberDepths[m] = 1;
          ++count;
          m = findMemberParent(tdec, m);
          assert(m >= 0);
        }
        for (m = member; count; --count)
        {
          memberDepths[m] += count;

#if defined(TU_DEBUG_TDEC)
          printf("        Member %d receives %d on subsequent path.\n", m, memberDepths[m]);
#endif /* TU_DEBUG_TDEC */
          newcolumn->reducedMembers[newcolumn->numReducedMembers].member = m;
          newcolumn->reducedMembers[newcolumn->numReducedMembers].depth = memberDepths[m];
          newcolumn->numReducedMembers++;
          m = findMemberParent(tdec, m);
        }
        if (memberDepths[m] < reducedRootDepth)
        {
          reducedRootMember = member;
          reducedRootDepth = memberDepths[m];
        }
      }
    }
    else
    {
#if defined(TU_DEBUG_TDEC)
      printf("      Edge %d is new.\n", edge);
#endif /* TU_DEBUG_TDEC */
    }
  }

#if defined(TU_DEBUG_TDEC)
  printf("      Root member is %d with temporary depth %d.\n", reducedRootMember, reducedRootDepth);
#endif /* TU_DEBUG_TDEC */

  for (int i = 0; i < newcolumn->numReducedMembers; ++i)
  {
#if defined(TU_DEBUG_TDEC)
    printf("        Shifting depth of %d: %d -> %d\n", newcolumn->reducedMembers[i].member,
      newcolumn->reducedMembers[i].depth,
      newcolumn->reducedMembers[i].depth + 1 - reducedRootDepth);
#endif /* TU_DEBUG_TDEC */
    newcolumn->reducedMembers[i].depth += 1 - reducedRootDepth;
  }

  qsort(newcolumn->reducedMembers, newcolumn->numReducedMembers, sizeof(ReducedMember),
    compareMemberDepths);

  for (int i = 0; i < newcolumn->numReducedMembers; ++i)
  {
    /* Remove members corresponding to non-positive depths since these are behind the root. */
    if (newcolumn->reducedMembers[i].depth <= 0)
      newcolumn->numReducedMembers = i;
#if defined(TU_DEBUG_TDEC)
    printf("        Member %d has depth %d.\n", newcolumn->reducedMembers[i].member,
      newcolumn->reducedMembers[i].depth);
#endif /* TU_DEBUG_TDEC */
  }
  assert(newcolumn->numReducedMembers > 0);
  assert(newcolumn->reducedMembers[0].member == reducedRootMember);

  /* We now create the mapping from members to reduced members. */
  for (int m = 0; m < tdec->numMembers; ++m)
    newcolumn->membersToReducedMembers[m] = NULL;
  for (int i = 0; i < newcolumn->numReducedMembers; ++i)
    newcolumn->membersToReducedMembers[newcolumn->reducedMembers[i].member] = &newcolumn->reducedMembers[i];

  TU_CALL( TUfreeStackArray(tu, &memberDepths) );

  return TU_OKAY;
}

static
TU_ERROR initializeReducedMemberEdgeLists(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  int* entryRows,               /**< Array of rows of new column's enries. */
  int numEntries                /**< Length of \p entryRows. */
)
{
#if defined(TU_DEBUG_TDEC)
  printf("    Initializing edge lists for members of reduced t-decomposition.\n");
#endif /* TU_DEBUG_TDEC */

  for (int v = 0; v < tdec->memNodes; ++v)
    newcolumn->nodesDegree[v] = 0;
  for (int e = 0; e < tdec->memEdges; ++e)
    newcolumn->edgesInPath[e] = false;

  /* (Re)allocate memory for edge lists. */
  assert(newcolumn->usedReducedEdgeStorage == 0);
  int requiredMemReducedEdgeStorage = numEntries;
  if (newcolumn->memReducedEdgeStorage < requiredMemReducedEdgeStorage)
  {
    newcolumn->memReducedEdgeStorage = 2 * requiredMemReducedEdgeStorage;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedEdgeStorage,
      newcolumn->memReducedEdgeStorage) );
  }

  /* Start with empty lists. */
  for (int i = 0; i < newcolumn->numReducedMembers; ++i)
    newcolumn->reducedMembers[i].firstReducedEdge = NULL;

  /* Fill edge lists. */
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    TU_TDEC_EDGE edge = (row < tdec->numRows) ? tdec->rowEdges[row].edge : -1;
    if (edge >= 0)
    {
      TU_TDEC_MEMBER member = tdec->edges[edge].member;
      assert(member >= 0);
      ReducedMember* reducedMember = newcolumn->membersToReducedMembers[member];
      newcolumn->reducedEdgeStorage[newcolumn->usedReducedEdgeStorage].next = reducedMember->firstReducedEdge;
      newcolumn->reducedEdgeStorage[newcolumn->usedReducedEdgeStorage].edge = edge;
      reducedMember->firstReducedEdge = &newcolumn->reducedEdgeStorage[newcolumn->usedReducedEdgeStorage];
      ++newcolumn->usedReducedEdgeStorage;

      newcolumn->edgesInPath[edge] = true;
      if (tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME)
      {
        newcolumn->nodesDegree[findEdgeHead(tdec, edge)]++;
        newcolumn->nodesDegree[findEdgeTail(tdec, edge)]++;
      }

#if defined(TU_DEBUG_TDEC)
      printf("      Edge %d <%d> belongs to reduced member %ld which is member %d.\n", edge,
        tdec->edges[edge].name,
        (reducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
        reducedMember->member);
#endif /* TU_DEBUG_TDEC */
    }
  }

  return TU_OKAY;
}

static
TU_ERROR computeReducedMemberChildren(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn  /**< new column. */
)
{
#if defined(TU_DEBUG_TDEC)
  printf("    Computing children of members of reduced t-decomposition.\n");
#endif /* TU_DEBUG_TDEC */

  /* Allocate memory for children of reduced members. */
  int requiredMemChildren = newcolumn->numReducedMembers;
  if (newcolumn->memChildrenStorage < requiredMemChildren)
  {
    newcolumn->memChildrenStorage = 2 * requiredMemChildren;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->childrenStorage, newcolumn->memChildrenStorage) );
  }

  /* Initialize numChildren to zero for all reduced members. */
  newcolumn->usedChildrenStorage = 0;
  for (int m = 0; m < newcolumn->numReducedMembers; ++m)
    newcolumn->reducedMembers[m].numChildren = 0;

  /* Count children of each reduced member. */
  for (int m = 1; m < newcolumn->numReducedMembers; ++m)
  {
    ReducedMember* parentReducedMember = newcolumn->membersToReducedMembers[
      tdec->members[newcolumn->reducedMembers[m].member].parentMember];

#if defined(TU_DEBUG_TDEC)
    printf("Reduced member %d (= member %d) has parent %ld (= member %d).\n",
      m, newcolumn->reducedMembers[m].member,
      (parentReducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
      tdec->members[newcolumn->reducedMembers[m].member].parentMember);
#endif /* TU_DEBUG_TDEC */
    parentReducedMember->numChildren++;
  }

  /* Set memory pointer of each reduced member. */
  for (int m = 0; m < newcolumn->numReducedMembers; ++m)
  {
    newcolumn->reducedMembers[m].children = &newcolumn->childrenStorage[newcolumn->usedChildrenStorage];
    newcolumn->usedChildrenStorage += newcolumn->reducedMembers[m].numChildren;
    newcolumn->reducedMembers[m].numChildren = 0;
  }

  /* Set children of each reduced member. */
  for (int m = 1; m < newcolumn->numReducedMembers; ++m)
  {
    ReducedMember* parentReducedMember = newcolumn->membersToReducedMembers[
      tdec->members[newcolumn->reducedMembers[m].member].parentMember];

#if defined(TU_DEBUG_TDEC)
    printf("Reduced member %ld (= member %d) has %d (= member %d) as child %d.\n",
      (parentReducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
      tdec->members[newcolumn->reducedMembers[m].member].parentMember,
      m, newcolumn->reducedMembers[m].member, parentReducedMember->numChildren);
#endif /* TU_DEBUG_TDEC */
    parentReducedMember->children[parentReducedMember->numChildren] = &newcolumn->reducedMembers[m];
    parentReducedMember->numChildren++;
  }

  return TU_OKAY;
}

/**
 * \brief Count the number of children of a reduced member having certain types.
 */

static
TU_ERROR countChildrenTypes(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  ReducedMember* reducedMember, /**< Reduced member. */
  int* pNumOneEnd,              /**< Number of children that (recursively) must contain one path end. */
  int* pNumTwoEnds              /**< Number of children that (recursively) must contain two path ends. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);

  if (pNumOneEnd)
    *pNumOneEnd = 0;
  if (pNumTwoEnds)
    *pNumTwoEnds = 0;

  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    ReducedMember* child = reducedMember->children[c];
    assert(child);
    if (pNumOneEnd && (child->type == TYPE_2_HEAD_END_TAIL_IN
      || child->type == TYPE_3_HEAD_END_TAIL_OUT
      || 0)) // child->type == TYPE_3_HEAD_OUT_TAIL_END))
    {
      (*pNumOneEnd)++;
    }
    else if (pNumTwoEnds && child->type == TYPE_4_HEAD_IN_TAIL_IN)
      (*pNumTwoEnds)++;
  }

  return TU_OKAY;
}

static
TU_ERROR determineTypes(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new-column structure. */
  ReducedMember* reducedMember  /**< Reduced member. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);

#if defined(TU_DEBUG_TDEC)
  printf("  determineTypes(reduced member %ld = member %d)\n",
    (reducedMember - &newcolumn->reducedMembers[0]) / sizeof(reducedMember),
    reducedMember->member);
#endif /* TU_DEBUG_TDEC */

  /* First handle children recursively. */
  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    TU_CALL( determineTypes(tu, tdec, newcolumn, reducedMember->children[c]) );

    /* Abort if some part indicates non-graphicness. */
    if (!newcolumn->remainsGraphic)
      return TU_OKAY;
  }

  bool isRoot = (reducedMember == &newcolumn->reducedMembers[0]);
  int numOneEnd;
  int numTwoEnds;
  TU_CALL( countChildrenTypes(tu, tdec, newcolumn, reducedMember, &numOneEnd, &numTwoEnds) );

#if defined(TU_DEBUG_TDEC)
  printf("    It has %d children with one end and %d with two ends.\n", numOneEnd,
    numTwoEnds);
#endif /* TU_DEBUG_TDEC */
  
  if (2*numTwoEnds + numOneEnd > 2)
  {
    newcolumn->remainsGraphic = false;
    return TU_OKAY;
  }

  /* Different behavior for bonds, polygons and prime components. */
  TU_TDEC_MEMBER member = reducedMember->member;
  if (tdec->members[member].type == TDEC_MEMBER_TYPE_BOND)
  {
    if (isRoot)
    {
      if (numTwoEnds + numOneEnd == 0)
      {
        
      }
      else
      {
        assert(0 == "Typing of root bond with path ends in children not implemented.");
      }
      reducedMember->type = TYPE_ROOT;
    }
    else
    {
      assert(0 == "Typing of non-root bond not implemented.");
    }
  }
  else if (tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON)
  {
    if (isRoot)
    {
      newcolumn->remainsGraphic = (numTwoEnds == 0) && (numOneEnd <= 2);
    }
    else
    {
      assert(0 == "Typing of non-root polygon not implemented.");
    }
  }
  else
  {
    assert(tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME);

    assert(0 == "Typing of prime not implemented.");
  }

  if (!isRoot && reducedMember->type == TYPE_1_HEAD_END_TAIL_END)
  {
    TU_TDEC_MEMBER parentMember = tdec->members[reducedMember->member].parentMember;
    ReducedMember* reducedParent = newcolumn->membersToReducedMembers[parentMember];
    TU_TDEC_EDGE markerOfParent = tdec->members[member].markerOfParent;

#if defined(TU_DEBUG_TDEC)
    printf("    Marker edge closes cycle.\n");
    printf("    Parent member %d is reduced member %ld.\n", parentMember,
      (reducedParent - newcolumn->reducedMembers) / sizeof(ReducedMember));
#endif /* TU_DEBUG_TDEC */

    /* Add marker edge of parent to reduced parent's reduced edges. */

    assert(newcolumn->usedReducedEdgeStorage < newcolumn->memReducedEdgeStorage);
    ReducedEdge* reducedEdge = &newcolumn->reducedEdgeStorage[newcolumn->usedReducedEdgeStorage];
    ++newcolumn->usedReducedEdgeStorage;
    reducedEdge->edge = markerOfParent;
    reducedEdge->next = reducedParent->firstReducedEdge;
    reducedParent->firstReducedEdge = reducedEdge;

    /* Indicate that marker edge of parent belongs to path. */
    newcolumn->edgesInPath[markerOfParent] = true;

    /* Increase node degrees of nodes in parent. */
    newcolumn->nodesDegree[findEdgeHead(tdec, markerOfParent)]++;
    newcolumn->nodesDegree[findEdgeTail(tdec, markerOfParent)]++;

#if defined(TU_DEBUG_TDEC)
    printf("    Added marker edge of parent to list of reduced edges.\n");
#endif /* TU_DEBUG_TDEC */
  }

  return TU_OKAY;
}

TU_ERROR TUtdecAddColumnCheck(TU* tu, TU_TDEC* tdec, TU_TDEC_NEWCOLUMN* newcolumn, int* entryRows,
  int numEntries)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(entryRows);
  assert(numEntries >= 1);

#if defined(TU_DEBUG_TDEC)
  printf("\n  Preparing to add a column with %d 1's.\n", numEntries);
#endif /* TU_DEBUG_TDEC */

  assertConsistency(tu, tdec);

  TU_CALL( initializeNewColumn(tu, tdec, newcolumn) );
  TU_CALL( findReducedDecomposition(tu, tdec, newcolumn, entryRows, numEntries) );
  TU_CALL( initializeReducedMemberEdgeLists(tu, tdec, newcolumn, entryRows, numEntries) );
  TU_CALL( computeReducedMemberChildren(tu, tdec, newcolumn) );
  TU_CALL( determineTypes(tu, tdec, newcolumn, &newcolumn->reducedMembers[0]) );

  if (newcolumn->remainsGraphic)
  {
#if defined(TU_DEBUG_TDEC)
    printf("    Adding the column would maintain graphicness.\n");
#endif /* TU_DEBUG_TDEC */
  }

  assertConsistency(tu, tdec);

  return TU_OKAY;
}

static
TU_ERROR addColumnPreprocessBond(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new-column structure. */
  ReducedMember* reducedMember, /**< Reduced member. */
  bool isRoot,                  /**< Whether \p reducedMember is the reduced root. */
  int* pNumAssignedTerminals    /**< Pointer to number of assigned terminal nodes. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);
  assert(pNumAssignedTerminals);

  int numOneEnd;
  int numTwoEnds;
  TU_CALL( countChildrenTypes(tu, tdec, newcolumn, reducedMember, &numOneEnd, &numTwoEnds) );
  
#if defined(TU_DEBUG_TDEC)
  printf("    addColumnPreprocessBond for reduced%s member %ld (member %d), #one-ends = %d, #two-ends = %d.\n",
    isRoot ? " root" : "", (reducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
    reducedMember->member, numOneEnd, numTwoEnds);
#endif /* TU_DEBUG_TDEC */

  if (numOneEnd == 0 && numTwoEnds == 0)
  {
    assert(reducedMember->firstReducedEdge);
    assert(*pNumAssignedTerminals == 0);

    newcolumn->terminalMember1 = reducedMember->member;
    newcolumn->terminalMember2 = reducedMember->member;
    *pNumAssignedTerminals = 2;

    return TU_OKAY;
  }

  assert(0 == "addColumnPreprocessBond is not implemented.");

  return TU_OKAY;
}

static
TU_ERROR addColumnPreprocessPrime(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new-column structure. */
  ReducedMember* reducedMember, /**< Reduced member. */
  bool isRoot,                  /**< Whether \p reducedMember is the reduced root. */
  int* pNumAssignedTerminals    /**< Pointer to number of assigned terminal nodes. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);
  assert(pNumAssignedTerminals);

  int numOneEnd;
  int numTwoEnds;
  TU_CALL( countChildrenTypes(tu, tdec, newcolumn, reducedMember, &numOneEnd, &numTwoEnds) );
  
#if defined(TU_DEBUG_TDEC)
  printf("    addColumnPreprocessPrime for reduced%s member %ld (member %d), #one-ends = %d, #two-ends = %d.\n",
    isRoot ? " root" : "", (reducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
    reducedMember->member, numOneEnd, numTwoEnds);
#endif /* TU_DEBUG_TDEC */

  assert(0 == "addColumnPreprocessPrime is not implemented.");

  return TU_OKAY;
}

/**
 * \brief Replaces an edge by a bond containing it.
 * 
 * The given member should have at least two edges.
 */

static
TU_ERROR createEdgeBond(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  TU_TDEC_MEMBER member,        /**< Polygon member to be squeezed. */
  TU_TDEC_EDGE edge,            /**< Edge. */
  TU_TDEC_EDGE* pChildEdge      /**< Pointer for storing the child marker edge to the new bond. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);
  assert(member < tdec->memMembers);
  assert(pChildEdge);

#if defined(TU_DEBUG_TDEC)
  printf("    Creating bond for edge %d in member %d.\n", edge, member);
#endif /* TU_DEBUG_TDEC */

  TU_TDEC_MEMBER bond = -1;
  TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_BOND, &bond) );
  tdec->members[bond].parentMember = member;
  tdec->members[bond].markerOfParent = edge;

  TU_TDEC_EDGE markerOfParent;
  TU_CALL( createMarkerEdge(tu, tdec, &markerOfParent, member, tdec->edges[edge].head,
    tdec->edges[edge].tail, true) );
  tdec->edges[markerOfParent].childMember = bond;
  tdec->edges[markerOfParent].next = tdec->edges[edge].next;
  tdec->edges[markerOfParent].prev = tdec->edges[edge].prev;
  assert(tdec->edges[markerOfParent].next != markerOfParent);
  tdec->edges[tdec->edges[markerOfParent].next].prev = markerOfParent;
  tdec->edges[tdec->edges[markerOfParent].prev].next = markerOfParent;
  if (tdec->members[member].firstEdge == edge)
    tdec->members[member].firstEdge = markerOfParent;
  tdec->members[bond].markerOfParent = markerOfParent;

  TU_TDEC_EDGE markerToParent;
  TU_CALL( createMarkerEdge(tu, tdec, &markerToParent, bond, -1, -1, false) );
  TU_CALL( addEdgeToMembersEdgeList(tu, tdec, markerToParent, bond) );
  tdec->members[bond].markerToParent = markerToParent;
  tdec->numMarkers++;

  TU_CALL( addEdgeToMembersEdgeList(tu, tdec, edge, bond) );

  *pChildEdge = markerOfParent;

  return TU_OKAY;
}

/**
 * \brief Squeezes subset of polygon edges into a new polygon connected via a bond.
 * 
 * Takes all edges of the polygon \p member for which \p edgesPredicate is the same as
 * \p predicateValue.
 */

static
TU_ERROR squeezePolygonEdges(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  TU_TDEC_MEMBER member,        /**< Polygon member to be squeezed. */
  bool* edgesPredicate,         /**< Map from edges to predicate. */
  bool predicateValue,          /**< Value of predicate. */
  TU_TDEC_EDGE* pChildEdge      /**< Pointer for storing the parent marker edge for the new bond. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);
  assert(member < tdec->memMembers);
  assert(edgesPredicate);
  assert(tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON);
  assert(pChildEdge);

#if defined(TU_DEBUG_TDEC)
  printf("    Squeezing polygon %d.\n", member);
#endif /* TU_DEBUG_TDEC */

  /* Initialize new polygon. */
  TU_TDEC_MEMBER polygon;
  TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_POLYGON, &polygon) );
  TU_TDEC_EDGE polygonParentMarker;
  TU_CALL( createMarkerEdge(tu, tdec, &polygonParentMarker, polygon, -1, -1, false) );
  TU_CALL( addEdgeToMembersEdgeList(tu, tdec, polygonParentMarker, polygon) );
  tdec->members[polygon].markerToParent = polygonParentMarker;

  /* Initialize new bond. */
  TU_TDEC_MEMBER bond;
  TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_BOND, &bond) );
  TU_TDEC_EDGE bondChildMarker;
  TU_CALL( createMarkerEdge(tu, tdec, &bondChildMarker, bond, -1, -1, true) );
  TU_CALL( addEdgeToMembersEdgeList(tu, tdec, bondChildMarker, bond) );
  tdec->numMarkers++;
  TU_TDEC_EDGE bondParentMarker;
  TU_CALL( createMarkerEdge(tu, tdec, &bondParentMarker, bond, -1, -1, false) );
  TU_CALL( addEdgeToMembersEdgeList(tu, tdec, bondParentMarker, bond) );
  tdec->members[polygon].markerOfParent = bondChildMarker;
  tdec->members[bond].markerToParent = bondParentMarker;

  /* Go through old polygon. */

  TU_TDEC_EDGE firstEdge = tdec->members[member].firstEdge;
  TU_TDEC_EDGE edge = firstEdge;
  do
  {
    /* Evaluate predicate. */
    bool value = edgesPredicate[edge];
    if ((value && !predicateValue) || (!value && predicateValue))
    {
#if defined(TU_DEBUG_TDEC)
      printf("        Edge %d <%d> does not satisfy the predicate.\n", edge,
        tdec->edges[edge].name);
      fflush(stdout);
#endif /* TU_DEBUG_TDEC */
      tdec->members[member].firstEdge = edge;
      edge = tdec->edges[edge].next;
      continue;
    }

#if defined(TU_DEBUG_TDEC)
    printf("        Edge %d = <%d> satisfies the predicate.\n", edge, tdec->edges[edge].name);
#endif /* TU_DEBUG_TDEC */

    assert(edge != tdec->members[member].markerToParent);

    /* Remove edge from old edge list. */
    TU_TDEC_EDGE oldPrev = tdec->edges[edge].prev;
    TU_TDEC_EDGE oldNext = tdec->edges[edge].next;
    tdec->edges[oldPrev].next = oldNext;
    tdec->edges[oldNext].prev = oldPrev;
    tdec->members[member].numEdges--;

    /* Add edge and newNode to new edge list. */
    TU_TDEC_EDGE newPrev = tdec->edges[polygonParentMarker].prev;
    tdec->edges[newPrev].next = edge;
    tdec->edges[polygonParentMarker].prev = edge;
    tdec->edges[edge].prev = newPrev;
    tdec->edges[edge].next = polygonParentMarker;

    edge = oldNext;
  }
  while (edge != firstEdge);

  /* Add child marker edge from old polygon to bond and add it to edge list. */
  TU_TDEC_EDGE memberChildMarker;
  TU_CALL( createMarkerEdge(tu, tdec, &memberChildMarker, member, -1, -1, true) );
  tdec->numMarkers++;
  tdec->members[bond].markerOfParent = memberChildMarker;
  TU_TDEC_EDGE oldPrev = tdec->edges[firstEdge].prev;
  tdec->edges[memberChildMarker].next = firstEdge;
  tdec->edges[memberChildMarker].prev = oldPrev;
  tdec->edges[oldPrev].next = memberChildMarker;
  tdec->edges[firstEdge].prev = memberChildMarker;

  /* Link all. */
  tdec->members[polygon].parentMember = bond;
  tdec->edges[bondChildMarker].childMember = polygon;
  tdec->members[bond].parentMember = member;
  tdec->edges[memberChildMarker].childMember = bond;

#if defined(TU_DEBUG_TDEC)
  printf("        Updated old polygon:\n");
  edge = firstEdge;
  do
  {
    printf("          Edge %d <%d>.\n", edge, tdec->edges[edge].name);
    edge = tdec->edges[edge].next;
  }
  while (edge != firstEdge);

  printf("        New polygon:\n");
  edge = polygonParentMarker;
  do
  {
    printf("          Edge %d <%d>.\n", edge, tdec->edges[edge].name);
    edge = tdec->edges[edge].next;
  }
  while (edge != polygonParentMarker);
#endif /* TU_DEBUG_TDEC */

  *pChildEdge = memberChildMarker;

  return TU_OKAY;
}


static
TU_ERROR addColumnPreprocessPolygon(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new-column structure. */
  ReducedMember* reducedMember, /**< Reduced member. */
  bool isRoot,                  /**< Whether \p reducedMember is the reduced root. */
  int* pNumAssignedTerminals    /**< Pointer to number of assigned terminal nodes. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);
  assert(pNumAssignedTerminals);

  int numOneEnd;
  int numTwoEnds;
  TU_CALL( countChildrenTypes(tu, tdec, newcolumn, reducedMember, &numOneEnd, &numTwoEnds) );

#if defined(TU_DEBUG_TDEC)
  printf("    addColumnPreprocessPolygon for reduced%s member %ld (member %d), #one-ends = %d, #two-ends = %d.\n",
    isRoot ? " root" : "", (reducedMember - newcolumn->reducedMembers) / sizeof(ReducedMember),
    reducedMember->member, numOneEnd, numTwoEnds);
#endif /* TU_DEBUG_TDEC */

  if (isRoot && numOneEnd == 0 && numTwoEnds == 0)
  {
    /* Root polygon containing both ends. */

    assert(reducedMember->firstReducedEdge);
    if (reducedMember->firstReducedEdge->next == NULL)
    {
      /* There is only one path edge, so we create a bond for that edge. */
      TU_TDEC_EDGE bondChildMarker;
      TU_CALL( createEdgeBond(tu, tdec, newcolumn, reducedMember->member,
        reducedMember->firstReducedEdge->edge, &bondChildMarker) );

      TU_TDEC_MEMBER bond = tdec->edges[bondChildMarker].childMember;
      newcolumn->terminalMember1 = bond;
      newcolumn->terminalMember2 = bond;
      *pNumAssignedTerminals = 2;
      return TU_OKAY;
    }
    else
    {
      /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
       * it to the remaining polygon. */

      TU_TDEC_EDGE bondChildMarker;
      TU_CALL( squeezePolygonEdges(tu, tdec, newcolumn, reducedMember->member,
        newcolumn->edgesInPath, true, &bondChildMarker) );

      TU_TDEC_MEMBER bond = tdec->edges[bondChildMarker].childMember;
      newcolumn->terminalMember1 = bond;
      newcolumn->terminalMember2 = bond;
      *pNumAssignedTerminals = 2;
      return TU_OKAY;
    }
  }

  assert(0 == "addColumnPreprocessPolygon is not implemented.");

  return TU_OKAY;
}

/**
 * \brief Preprocessing of reduced t-decomposition before the actual modification.
 * 
 * Processes the reduced members in depth-first search manner and does the following:
 * - Polygons are squeezed.
 * - Terminal nodes and (reduced) members are detected.
 */

static
TU_ERROR addColumnPreprocess(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new-column structure. */
  ReducedMember* reducedMember, /**< Reduced member. */
  int* pNumAssignedTerminals    /**< Pointer to number of assigned terminal nodes. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);

#if defined(TU_DEBUG_TDEC)
  printf("  addColumnPreprocess(reduced member %ld = member %d)\n",
    (reducedMember - &newcolumn->reducedMembers[0]) / sizeof(reducedMember),
    reducedMember->member);
#endif /* TU_DEBUG_TDEC */

  assertConsistency(tu, tdec);

  /* Handle children recursively. */
  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    TU_CALL( addColumnPreprocess(tu, tdec, newcolumn, reducedMember->children[c],
      pNumAssignedTerminals) );
  }

  bool isRoot = (reducedMember == &newcolumn->reducedMembers[0]);

  /* Different behavior for bonds, polygons and prime components. */

  if (tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_BOND)
  {
    TU_CALL( addColumnPreprocessBond(tu, tdec, newcolumn, reducedMember, isRoot,
      pNumAssignedTerminals) );
  }
  else if (tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_PRIME)
  {
    TU_CALL( addColumnPreprocessPrime(tu, tdec, newcolumn, reducedMember, isRoot,
      pNumAssignedTerminals) );
  }
  else
  {
    assert(tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_POLYGON);

    TU_CALL( addColumnPreprocessPolygon(tu, tdec, newcolumn, reducedMember, isRoot,
      pNumAssignedTerminals) );
  }

  if (reducedMember->type == TYPE_1_HEAD_END_TAIL_END)
  {
    ReducedMember* parent = newcolumn->membersToReducedMembers[findMember(tdec,
      tdec->members[reducedMember->member].markerOfParent)];
    assert(parent);

    if (newcolumn->usedReducedEdgeStorage == newcolumn->memReducedEdgeStorage)
    {
      newcolumn->memReducedEdgeStorage *= 2;
      TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedEdgeStorage,
        newcolumn->memReducedEdgeStorage) );
      for (int e = newcolumn->usedChildrenStorage; e < newcolumn->memChildrenStorage; ++e)
        newcolumn->reducedEdgeStorage[e].next = &newcolumn->reducedEdgeStorage[e+1];
      
    }

    assert(0 == "Not tested.");
  }

  assertConsistency(tu, tdec);

#if 0
  
    if (isRoot)
    {
      if (*pNumAssignedTerminals == 0)
      {
        newcolumn->terminalMember1 = reducedMember;
        newcolumn->terminalNode1 = findEdgeHead(tdec, reducedMember->firstReducedEdge->edge);
        (*pNumAssignedTerminals)++;
      }
      if (*pNumAssignedTerminals == 1)
      {
        newcolumn->terminalMember2 = reducedMember;
        newcolumn->terminalNode2 = findEdgeTail(tdec, reducedMember->firstReducedEdge->edge);
        (*pNumAssignedTerminals)++;
      }
    }
    else
    {
      assert(0 == "addColumnPreprocess for non-root bond not yet implemented.");
    }
  }
  else if (tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_POLYGON)
  {
#if defined(TU_DEBUG_TDEC)
  printf("    Member is a polygon.\n");
#endif /* TU_DEBUG_TDEC */

    /* The parent marker edge p (unless root). */
    TU_TDEC_EDGE markerToParent = tdec->members[reducedMember->member].markerToParent;
    TU_TDEC_EDGE start, edge;

    /* The child marker edges c1, c2 for the ends of the path. */
//     ReducedMember* endChild[2] = {NULL, NULL};
    TU_TDEC_EDGE endChildEdge[2] = {-1, -1};
//     bool endChildHead[2] = {false, false};
    for (int c = 0; c < reducedMember->numChildren; ++c)
    {
      Type type = reducedMember->children[c]->type;
      if (type == TYPE_2_HEAD_END_TAIL_IN || type == TYPE_3_HEAD_END_TAIL_OUT)
      {
        int childNum = endChildEdge[0] >= 0 ? 1 : 0;
        assert(endChildEdge[childNum] < 0);
//         endChild[childNum] = reducedMember->children[c];
        endChildEdge[childNum] = tdec->members[reducedMember->children[c]->member].markerOfParent;
//         endChildHead[childNum] = (type == TYPE_2_HEAD_END_TAIL_IN || type == TYPE_3_HEAD_END_TAIL_OUT );
      }
    }

    assert(endChildEdge[0] == -1 || 0 == "Not implemented: addColumnPreprocess with path-end children.");

    /* Reorder the edge list of the member as follows:
     *
     * [child-marker 0]
     * regular edge
     * regular edge
     * ...
     * regular edge
     * [parent marker]
     * [child-marker 1]
     * 
     * the last two are interchanged in case of a root.
     */

    /* Swap child-marker 1 with edge after parent marker. */
    if (endChildEdge[1] >= 0)
    {
      TU_CALL( swapEdges(tu, tdec, reducedMember->member, tdec->edges[markerToParent].next,
        endChildEdge[1]) );
    }

    /* Swap child marker 1 with parent marker if root. */
    if (endChildEdge[1] >= 0 && isRoot)
    {
      TU_CALL( swapEdges(tu, tdec, reducedMember->member, markerToParent, endChildEdge[1]) );
      edge = endChildEdge[1];
    }
    else
    {
      edge = markerToParent;
    }

    /* Swap regular edges. */
    for (ReducedEdge* reducedEdge = reducedMember->firstReducedEdge; reducedEdge;
      reducedEdge = reducedEdge->next)
    {
      TU_CALL( swapEdges(tu, tdec, reducedMember->member, tdec->edges[edge].prev,
        reducedEdge->edge) );
      edge = reducedEdge->edge;
    }

    /* Swap child-marker 0 with edge before last reduced edge. */
    if (endChildEdge[0] >= 0)
    {
      edge = tdec->edges[edge].prev;
      TU_CALL( swapEdges(tu, tdec, reducedMember->member, edge,
        endChildEdge[0]) );
    }

    /* Make this edge the first in the member's list. */
    tdec->members[reducedMember->member].firstEdge = edge;

#if defined(TU_DEBUG_TDEC)
    start = (endChildEdge[1] >= 0) ? tdec->edges[markerToParent].next : markerToParent;
    edge = start;
    do
    {
      if (edge == endChildEdge[0])
        printf("      1st end ");
      else if (edge == endChildEdge[1])
        printf("      2nd end ");
      else if (edge == markerToParent)
        printf("      Parent marker ");
      else
        printf("      Regular ");
      if (edge == tdec->members[reducedMember->member].firstEdge)
        printf("(first of this member) ");
      printf("edge %d <%d> {%d,%d}\n", edge, tdec->edges[edge].name, findEdgeHead(tdec, edge),
        findEdgeTail(tdec, edge));
      edge = tdec->edges[edge].prev;
    }
    while  (edge != start);
#endif /* TU_DEBUG_TDEC */

    if (isRoot)
    {
      if (*pNumAssignedTerminals == 0)
      {
        /* Ends of path in this polygon. */

        TU_TDEC_EDGE firstEdge = tdec->members[reducedMember->member].firstEdge;
        TU_TDEC_NODE head1 = findEdgeHead(tdec, firstEdge);
        TU_TDEC_NODE tail1 = findEdgeTail(tdec, firstEdge);
        TU_TDEC_NODE head2 = findEdgeHead(tdec, tdec->edges[firstEdge].next);
        TU_TDEC_NODE tail2 = findEdgeTail(tdec, tdec->edges[firstEdge].next);
        newcolumn->terminalNode1 = (head1 == head2 || head1 == tail2) ? tail1 : head1;

#if defined(TU_DEBUG_TDEC)
        printf("      First node of path is %d.\n", newcolumn->terminalNode1);
#endif /* TU_DEBUG_TDEC */

        TU_TDEC_EDGE lastEdge = tdec->edges[tdec->members[reducedMember->member].markerToParent].prev;
        head1 = findEdgeHead(tdec, lastEdge);
        tail1 = findEdgeTail(tdec, lastEdge);
        head2 = findEdgeHead(tdec, tdec->edges[lastEdge].prev);
        tail2 = findEdgeTail(tdec, tdec->edges[lastEdge].prev);
        newcolumn->terminalNode2 = (head1 == head2 || head1 == tail2) ? tail1 : head1;

#if defined(TU_DEBUG_TDEC)
        printf("      Last node of path is %d.\n", newcolumn->terminalNode2);
#endif /* TU_DEBUG_TDEC */

        newcolumn->terminalMember1 = reducedMember;
        newcolumn->terminalMember2 = reducedMember;
        (*pNumAssignedTerminals) = 2;
      }
      else
      {
        assert(0 == "addColumnPreprocess for root polygon with end-children not yet implemented.");
      }
    }
    else
    {
      assert(0 == "addColumnPreprocess for non-root polygon not yet implemented.");
    }
  }
  else
  {
    assert(tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_PRIME);

#if defined(TU_DEBUG_TDEC)
    printf("    Member is prime.\n");
#endif /* TU_DEBUG_TDEC */

    if (isRoot)
    {
      assert(0 == "addColumnPreprocess for root prime not yet implemented.");
    }
    else
    {
      assert(0 == "addColumnPreprocess for non-root prime not yet implemented.");
    }
  }

#endif

  return TU_OKAY;
}

static
TU_ERROR createNewRowsPolygon(
  TU* tu,                   /**< \ref TU environment. */
  TU_TDEC* tdec,            /**< t-decomposition. */
  TU_TDEC_MEMBER* pmember,  /**< Pointer for storing the new member or -1 if there is none. */
  TU_TDEC_EDGE* pedge,      /**< Pointer for storing the new edge. */
  TU_TDEC_NODE head,        /**< Head node. */
  TU_TDEC_NODE tail,        /**< Tail node. */
  int column,               /**< Index of new column to be added. */
  int* entryRows,           /**< Array of rows with 1-entry in this column. */
  int numEntries            /**< Number of 1-entries in this column. */
)
{
  assert(tu);
  assert(tdec);
  assert(column >= 0);
  assert(entryRows);
  assert(numEntries >= 0);
  assert(pmember);
  assert(pedge);

#if defined(TU_DEBUG_TDEC)
  printf("      Creating polygon for new rows.\n");
#endif /* TU_DEBUG_TDEC */
  
  /* Count new rows. */
  int countNewRows = 0;
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    if (row >= tdec->numRows || tdec->rowEdges[row].edge < 0)
      ++countNewRows;
  }

  if (countNewRows)
  {
#if defined(TU_DEBUG_TDEC)
    printf("      There are %d new rows.\n", countNewRows);
#endif /* TU_DEBUG_TDEC */

    /*
     * newEdge = parent marker edge.
     *
     * markerEdge ---------------> cotreeEdge
     *       ^                         |
     *       |                         |
     * first tree edge <- ... <- last tree edge
     *
     * Arrow e --> f means that e->next = f, f->prev = e, e->head = f->tail.
     */

    TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_POLYGON, pmember) );
    TU_TDEC_MEMBER newMember = *pmember;
    TU_TDEC_EDGE parentMarkerEdge;
    TU_CALL( createMarkerEdge(tu, tdec, &parentMarkerEdge, INT_MIN, head, tail, true) );
    tdec->edges[parentMarkerEdge].childMember = newMember;

    /* Add child marker edge and link it to marker edges. */
    TU_TDEC_EDGE childMarkerEdge;
    TU_CALL( createMarkerEdge(tu, tdec, &childMarkerEdge, newMember, -1, -1, false) );
    TU_CALL( addEdgeToMembersEdgeList(tu, tdec, childMarkerEdge, newMember) );
    tdec->members[newMember].markerOfParent = parentMarkerEdge;
    tdec->members[newMember].markerToParent = childMarkerEdge;
    tdec->numMarkers++;

    /* Add new tree edges. */
    for (int p = 0; p < numEntries; ++p)
    {
      int row = entryRows[p];
      if (row >= tdec->numRows || tdec->rowEdges[row].edge < 0)
      {
        TU_TDEC_EDGE treeEdge;
        TU_CALL( createRowEdge(tu, tdec, &treeEdge, newMember, -1, -1, row) );
        TU_CALL( addEdgeToMembersEdgeList(tu, tdec, treeEdge, newMember) );
      }
    }

    /* Add cotree edge. */
    TU_TDEC_EDGE cotreeEdge;
    TU_CALL( createColumnEdge(tu, tdec, &cotreeEdge, newMember, -1, -1, column) );
    TU_CALL( addEdgeToMembersEdgeList(tu, tdec, cotreeEdge, newMember) );

    *pedge = parentMarkerEdge;
  }
  else
  {
    *pmember = -1;
    TU_CALL( createColumnEdge(tu, tdec, pedge, INT_MIN, head, tail, column) );
  }

  return TU_OKAY;
}

TU_ERROR TUtdecAddColumnApply(TU* tu, TU_TDEC* tdec, TU_TDEC_NEWCOLUMN* newcolumn, int column,
  int* entryRows, int numEntries)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(newcolumn->remainsGraphic);
  assert(newcolumn->numReducedMembers > 0);

#if defined(TU_DEBUG_TDEC)
  printf("  Adding a column with %d 1's.\n", numEntries);
#endif /* TU_DEBUG_TDEC */

  assertConsistency(tu, tdec);

  int numAssignedTerminals = 0;
  TU_CALL( addColumnPreprocess(tu, tdec, newcolumn, &newcolumn->reducedMembers[0],
    &numAssignedTerminals) );
  assert(numAssignedTerminals == 2);

#if defined(TU_DEBUG_TDEC)
  printf("    Preprocessing done. Reduced terminal members are %d and %d.\n",
    newcolumn->terminalMember1, newcolumn->terminalMember2);
  fflush(stdout);
#endif /* TU_DEBUG_TDEC */  

  TU_TDEC_MEMBER newMember;
  TU_TDEC_EDGE newEdge;
  TU_CALL( createNewRowsPolygon(tu, tdec, &newMember, &newEdge, newcolumn->terminalNode1,
    newcolumn->terminalNode2, column, entryRows, numEntries) );
#if defined(TU_DEBUG_TDEC)
  printf("    New edge is %d.\n", newEdge);
  fflush(stdout);
#endif /* TU_DEBUG_TDEC */

  ReducedMember* rootReduced = &newcolumn->reducedMembers[0];
  TU_TDEC_MEMBER rootMember = rootReduced->member;
  assert(rootReduced);

  if (newcolumn->terminalMember1 == newcolumn->terminalMember2)
  {
    if (tdec->members[newcolumn->terminalMember1].type == TDEC_MEMBER_TYPE_BOND)
    {
      /* Add edge to the bond.  */

      tdec->edges[newEdge].member = newcolumn->terminalMember1;
      tdec->edges[newEdge].head = newcolumn->terminalNode1;
      tdec->edges[newEdge].tail = newcolumn->terminalNode2;
      TU_CALL( addEdgeToMembersEdgeList(tu, tdec, newEdge, newcolumn->terminalMember1) );
      if (newMember >= 0)
      {
        tdec->members[newMember].parentMember = rootMember;
      }
    }
    else if (tdec->members[newcolumn->terminalMember1].type == TDEC_MEMBER_TYPE_POLYGON)
    {
      assert(0 == "Adding of column with same end component of type polygon not implemented.");
    }
    else
    {
      assert(tdec->members[newcolumn->terminalMember1].type == TDEC_MEMBER_TYPE_PRIME);

      assert(0 == "Adding of column with same end component of type prime not implemented.");
    }
  }
  else
  {
    assert(0 == "Adding of column with different end components not implemented.");
  }

  assertConsistency(tu, tdec);

  return TU_OKAY;
}

TU_ERROR testGraphicnessTDecomposition(TU* tu, TU_CHRMAT* matrix, TU_CHRMAT* transpose,
  bool* isGraphic, TU_GRAPH* graph, TU_GRAPH_EDGE* basis, TU_GRAPH_EDGE* cobasis,
  TU_SUBMAT** psubmatrix)
{
  assert(tu);
  assert(matrix);
  assert(transpose);
  assert(!graph || (TUgraphNumNodes(graph) == 0 && TUgraphNumEdges(graph) == 0));
  assert(!psubmatrix || !*psubmatrix);
  assert(!basis || graph);
  assert(!cobasis || graph);

#if defined(TU_DEBUG_TDEC)
  printf("testGraphicnessTDecomposition called for a 1-connected %dx%d matrix.\n",
    matrix->numRows, matrix->numColumns);
  TUchrmatPrintDense(stdout, (TU_CHRMAT*) matrix, ' ', true);
#endif /* TU_DEBUG_TDEC */

  if (matrix->numNonzeros == 0)
  {
    *isGraphic = true;
    if (graph)
    {
      /* Construct a path with numRows edges and with numColumns loops at 0. */

      TU_GRAPH_NODE s;
      TU_CALL( TUgraphAddNode(tu, graph, &s) );
      for (int c = 0; c < matrix->numColumns; ++c)
      {
        TU_GRAPH_EDGE e;
        TU_CALL( TUgraphAddEdge(tu, graph, s, s, &e) );
        if (cobasis)
          *cobasis++ = e;
      }
      for (int r = 0; r < matrix->numRows; ++r)
      {
        TU_GRAPH_NODE t;
        TU_CALL( TUgraphAddNode(tu, graph, &t) );
        TU_GRAPH_EDGE e;
        TU_CALL( TUgraphAddEdge(tu, graph, s, t, &e) );
        if (basis)
          *basis++ = e;
        s = t;
      }

#if defined(TU_DEBUG_TDEC)
      printf("Constructed graph with %d nodes and %d edges.\n", TUgraphNumNodes(graph),
        TUgraphNumEdges(graph));
#endif /* TU_DEBUG_TDEC */
    }
    return TU_OKAY;
  }

  int rootRow = transpose->entryColumns[0];
#if defined(TU_DEBUG_TDEC)
  printf("  Root row is %d.\n", rootRow);
#endif /* TU_DEBUG_TDEC */
  TU_TDEC* tdec = NULL;
  TU_CALL( TUtdecCreate(tu, &tdec, rootRow, 0, 0, 0, 0, 0) ); /* TODO: avoid reallocations. */

  /* Process each column. */
  TU_TDEC_NEWCOLUMN* newcol = NULL;
  TUtdecnewcolumnCreate(tu, &newcol);
  for (int column = 0; column < matrix->numColumns; ++column)
  {
    TUtdecAddColumnCheck(tu, tdec, newcol,
      &transpose->entryColumns[transpose->rowStarts[column]],
      transpose->rowStarts[column+1] - transpose->rowStarts[column]);

    if (newcol->remainsGraphic)
    {
      TUtdecAddColumnApply(tu, tdec, newcol, column,
        &transpose->entryColumns[transpose->rowStarts[column]],
        transpose->rowStarts[column+1] - transpose->rowStarts[column]);
    }
    else
    {
      *isGraphic = false;
      assert(!"Not implemented");
    }
  }
  TUtdecnewcolumnFree(tu, &newcol);

  if (*isGraphic && graph)
  {
    TUtdecToGraph(tu, tdec, graph, true, basis, cobasis, NULL);
  }

  TUtdecFree(tu, &tdec);

  return TU_OKAY;
}

