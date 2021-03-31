#define TU_DEBUG /* Uncomment to enable general debugging. */
#define TU_DEBUG_TDEC /* Uncomment to enable debugging of t-decompositions. */
// #define TU_DEBUG_TDEC_SQUEEZE /* Uncomment to enable debug output for squeezing of polygons. */
#define TU_DEBUG_DOT /* Uncomment to output dot files after modifications of the t-decomposition. */

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
  int name;                   /**< \brief Name of this edge.
                               *
                               * 0, 1, ..., m-1 indicate rows, -1,-2, ..., -n indicate columns,
                               * and for (small) k >= 0, MAX_INT-k and -MAX_INT+k indicate
                               * markers of the parent and to the parent, respectively. */
  TU_TDEC_MEMBER member;      /**< \brief Member this edge belongs to or -1 if in free list. */
  TU_TDEC_NODE head;          /**< \brief Head node of this edge for a prime member, -1 otherwise. */
  TU_TDEC_NODE tail;          /**< \brief Tail node of this edge for a prime member, -1 otherwise. */
  TU_TDEC_EDGE prev;          /**< \brief Next edge of this member. */
  TU_TDEC_EDGE next;          /**< \brief Previous edge of this member. */
  TU_TDEC_MEMBER childMember; /**< \brief Child member linked to this edge, or -1. */
} TU_TDEC_EDGE_DATA;

typedef struct
{
  TU_TDEC_MEMBER_TYPE type;             /**< \brief Type of member. Only valid for representative member. */
  TU_TDEC_MEMBER representativeMember;  /**< \brief Representative of member, or -1 if this is a representative member. */
  TU_TDEC_MEMBER parentMember;          /**< \brief Parent member of this member or -1 for a root. Only valid for representative member. */
  int numEdges;                         /**< \brief Number of edges. Only valid for representative member. */
  TU_TDEC_EDGE markerToParent;          /**< \brief Parent marker edge. Only valid for representative member. */
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

/**
 * \brief Checks whether \p tdec has consistent edge data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
char* consistencyEdges(
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
    int countEdges = 0;
    if (edge >= 0)
    {
      do
      {
        if (edge < 0 || edge >= tdec->memEdges)
          return TUconsistencyMessage("edge %d of member %d out of range.", member, edge);
        if (tdec->edges[edge].next < 0 || tdec->edges[edge].next > tdec->memEdges)
          return TUconsistencyMessage("edge %d of member %d has next out of range", member, edge);
        if (tdec->edges[tdec->edges[edge].next].prev != edge)
          return TUconsistencyMessage("member %d has inconsistent edge list", member);
        if (findEdgeMember(tdec, edge) != member)
          return TUconsistencyMessage("edge %d belongs to member %d but is in member %d's edge list.", edge,
            findEdgeMember(tdec, edge), member);
        edge = tdec->edges[edge].next;
        countEdges++;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    if (countEdges != tdec->members[member].numEdges)
    {
      return TUconsistencyMessage("member %d has %d edges, but numEdges %d", member, countEdges,
        tdec->members[member].numEdges);
    }
  }

  return NULL;
}

/**
 * \brief Checks whether \p tdec has consistent member data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
char* consistencyMembers(
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
      return TUconsistencyMessage("member %d has invalid type", member);
  }

  return NULL;
}

/**
 * \brief Checks whether \p tdec has consistent node data.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
char* consistencyNodes(
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
          return TUconsistencyMessage("edge %d of prime member %d has invalid head node", edge, member);
        if (tail < 0)
          return TUconsistencyMessage("edge %d of prime member %d has invalid tail node", edge, member);
        if (head >= tdec->memNodes)
          return TUconsistencyMessage("edge %d of prime member %d has head node out of range", edge, member);
        if (tail >= tdec->memNodes)
          return TUconsistencyMessage("edge %d of prime member %d has tail node out of range", edge, member);
      }
      edge = tdec->edges[edge].next;
    }
    while (edge != tdec->members[member].firstEdge);
  }

  return NULL;
}

/**
 * \brief Checks whether the members of \p tdec form a forest.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
char* consistencyTree(
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

    int length = 0;
    TU_TDEC_MEMBER current;
    for (current = tdec->members[member].parentMember; current >= 0; current = tdec->members[current].parentMember)
    {
      ++length;
      if (length > tdec->numMembers)
        return "infinite member parent loop";
    }
  }

  return NULL;
}

typedef enum
{
  TYPE_UNKNOWN = 0,
  TYPE_1_CLOSES_CYCLE = 1,        /**< Parent marker edge plus path is a cycle. */
  TYPE_2_SHORTCUT = 2,            /**< One node of the parent marker edge is a path end and the other is an inner node. */
  TYPE_3_EXTENSION = 3,           /**< One node of the parent marker edge is a path end and the other does not belong to the path. */
  TYPE_4_CONNECTS_TWO_PATHS = 4,  /**< Adding the parent marker edge connects two paths to a single one. */
  TYPE_5_ROOT = 5,                /**< Root member. */
  TYPE_6_OTHER = 6                /**< All other cases. */
} Type;

/**
 * \brief Additional edge information specific to a path.
 */

typedef struct _PathEdge
{
  TU_TDEC_EDGE edge;          /**< \brief The edge in the t-decomposition. */
  struct _PathEdge* next;  /**< \brief Next edge of this reduced member, or \c NULL. */
} PathEdge;

/**
 * \brief Additional member information specfic to a given path.
 * 
 * @TODO: Maybe add parent reduced member as well, so we don't have to go via the membersToReducedMembers array.
 */

typedef struct _ReducedMember
{
  TU_TDEC_MEMBER member;            /**< \brief The member from the t-decomposition. */
  TU_TDEC_MEMBER rootMember;        /**< \brief The root member of this component of the t-decomposition. */
  int depth;                        /**< \brief Depth of this member in the reduced t-decomposition. */
  Type type;                        /**< \brief Type of this member. */
  struct _ReducedMember* parent;    /**< \brief Parent in the reduced t-decomposition. */
  int numChildren;                  /**< \brief Number of children in the reduced t-decomposition. */
  struct _ReducedMember** children; /**< \brief Children in the reduced t-decomposition. */
  PathEdge* firstPathEdge;          /**< \brief First edge in linked list of path edges of \p member. */
  TU_TDEC_EDGE polygonPathEdge;     /**< \brief For polygons, edge representing squeezed off path polygon. */
  TU_TDEC_EDGE polygonNonpathEdge;  /**< \brief For polygons, edge representing squeezed off non-path polygon. */
  TU_TDEC_NODE primeEndNodes[4];    /**< \brief For primes, the end nodes of the paths inside the member (or -1). */
} ReducedMember;

typedef struct _ReducedComponent
{
  int rootDepth;                    /**< \brief Depth of reduced root member. */
  ReducedMember* root;              /**< \brief Reduced root member. */
  TU_TDEC_NODE terminalNode[2];     /**< \brief Terminal nodes of path. */
  TU_TDEC_MEMBER terminalMember[2]; /**< \brief Terminal members of path. */
  int numTerminals;
} ReducedComponent;

struct _TU_TDEC_NEWCOLUMN
{
  bool remainsGraphic;                      /**< \brief Indicator whether adding this column maintains graphicness. */
  int memReducedMembers;                    /**< \brief Allocated memory for \c reducedMembers. */
  int numReducedMembers;                    /**< \brief Number of members in \c reducedMembers. */
  ReducedMember* reducedMembers;            /**< \brief Array of reduced members, sorted by increasing depth. */
  ReducedMember** membersToReducedMembers;  /**< \brief Array mapping members to members of the reduced t-decomposition. */

  ReducedComponent* reducedComponents;      /**< \brief Array with reduced root members. */
  int memReducedComponents;                 /**< \brief Allocated memory for \c reducedComponents. */
  int numReducedComponents;                 /**< \brief Number of reduced root members. */

  PathEdge* pathEdgeStorage;             /**< \brief Storage for edge lists of reduced members. */
  int memPathEdges;                /**< \brief Allocated memory for \c reducedEdgeStorage. */
  int numPathEdges;               /**< \brief Number of stored edges in \c reducedEdgeStorage. */

  ReducedMember** childrenStorage;          /**< \brief Storage for members' arrays of children in reduced t-decomposition. */
  int usedChildrenStorage;                  /**< \brief Number of stored children in \c childrenStorage. */
  int memChildrenStorage;                   /**< \brief Allocated memory for \c childrenStorage. */

  int* nodesDegree;                         /**< \brief Map from nodes to degree w.r.t. path edges. */
  bool* edgesInPath;                        /**< \brief Map from edges to indicator for being in the path. */
};

int compareMemberDepths(const void* a, const void* b)
{
  const ReducedMember* first = a;
  const ReducedMember* second = b;
  /* Negative depths are moved to the end. */
  if (first->depth < 0)
    return +1;
  if (second->depth < 0)
    return -1;
  return first->depth - second->depth;
}

/**
 * \brief Checks whether \p tdec has consistent parent/child structure of members.
 * 
 * \returns Explanation of inconsistency, or \c NULL.
 */

static
char* consistencyParentChild(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  if (tdec->memMembers < tdec->numMembers)
    return TUconsistencyMessage("member count and memory are inconsistent");
  if (tdec->numMembers < 0)
    return TUconsistencyMessage("negative member count");

  int* countChildren = NULL;
  if (TUallocStackArray(tu, &countChildren, tdec->memMembers) != TU_OKAY)
    return TUconsistencyMessage("stack allocation in consistencyParentChild() failed");
  for (int m = 0; m < tdec->memMembers; ++m)
    countChildren[m] = 0;

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (!isRepresentativeMember(tdec, member))
      continue;

    if (tdec->members[member].parentMember >= tdec->memMembers)
    {
      TUfreeStackArray(tu, &countChildren);
      return TUconsistencyMessage("parent member of %d is out of range", member);
    }
    if (tdec->members[member].parentMember >= 0)
      countChildren[tdec->members[member].parentMember]++;
  }

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (!isRepresentativeMember(tdec, member))
      continue;

    TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
    if (edge < 0)
      continue;
    do
    {
      if (tdec->edges[edge].childMember >= 0)
      {
        countChildren[member]--;

        if (findMember(tdec, tdec->members[findMember(tdec, tdec->edges[edge].childMember)].parentMember) != findMember(tdec, member))
        {
          TUfreeStackArray(tu, &countChildren);
          return TUconsistencyMessage("member %d has child edge %d for child %d whose parent member is %d",
            member, edge, findMember(tdec, tdec->edges[edge].childMember),
            findMember(tdec, tdec->members[findMember(tdec, tdec->edges[edge].childMember)].parentMember));
        }
        if (tdec->members[findMember(tdec, tdec->edges[edge].childMember)].markerOfParent != edge)
        {
          TUfreeStackArray(tu, &countChildren);
          return TUconsistencyMessage("member %d has child edge %d for child %d whose parent's markerOfParent is %d",
            member, edge, findMember(tdec, tdec->edges[edge].childMember),
            tdec->members[findMember(tdec, tdec->edges[edge].childMember)].markerOfParent);
        }
        TU_TDEC_EDGE markerChild = tdec->members[findMember(tdec, tdec->edges[edge].childMember)].markerToParent;
        if (tdec->edges[markerChild].name != -tdec->edges[edge].name)
        {
          TUfreeStackArray(tu, &countChildren);
          return TUconsistencyMessage("marker edges %d and %d of members %d (parent) and %d (child) have names %d and %d.",
            edge, markerChild, member, findEdgeMember(tdec, markerChild), tdec->edges[edge].name,
            tdec->edges[markerChild].name);
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

char* TUtdecConsistency(TU* tu, TU_TDEC* tdec)
{
  char* message = NULL;
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
  assert(tdec);
  assert(edge >= 0);
  assert(edge < tdec->memEdges);
  assert(tdec->edges[edge].head >= 0);
  assert(tdec->edges[edge].head < tdec->memNodes);
  return findNode(tdec, tdec->edges[edge].head);
}

static // TODO: inline
TU_TDEC_NODE findEdgeTail(TU_TDEC* tdec, TU_TDEC_EDGE edge)
{
  assert(tdec);
  assert(edge >= 0);
  assert(edge < tdec->memEdges);
  assert(tdec->edges[edge].tail >= 0);
  assert(tdec->edges[edge].tail < tdec->memNodes);
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
    printf("            createNode returns free node %d.\n", node);
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
    TUdbgMsg(12, "createNode enlarges node array to %d and returns node %d.\n", newSize, node);
  }
  tdec->nodes[node].representativeNode = -1;
  tdec->numNodes++;

  *pnode = node;

  return TU_OKAY;
}

/**
 * \brief Adds \p edge to the edge list of \p member.
 * 
 * Requires that \p edge already has \p member as its member.
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
  assert(tdec->edges[edge].member == member);

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
 * \brief Removes \p edge from the edge list of \p member.
 */

static
TU_ERROR removeEdgeFromMembersEdgeList(
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
  assert(tdec->edges[edge].member == member);

  if (tdec->members[member].numEdges == 1)
    tdec->members[member].firstEdge = -1;
  else
  {
    if (tdec->members[member].firstEdge == edge)
      tdec->members[member].firstEdge = tdec->edges[edge].next;

    assert(tdec->members[member].firstEdge != edge);

    tdec->edges[tdec->edges[edge].prev].next = tdec->edges[edge].next;
    tdec->edges[tdec->edges[edge].next].prev = tdec->edges[edge].prev;
  }

  tdec->members[member].numEdges--;

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
    TUdbgMsg(12, "Creating edge %d by using a free edge.\n", edge);
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
    TUdbgMsg(12, "Creating edge %d and reallocating edge to array %d elements.\n", edge, newSize);
  }

  tdec->edges[edge].member = member;
  tdec->numEdges++;

  *pedge = edge;

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
    data->name = -INT_MAX + tdec->numMarkers;
  TUdbgMsg(12, "Created %s marker edge {%d,%d} of member %d.\n", isParent ? "parent" : "child", head, tail, member);

  return TU_OKAY;
}

static
TU_ERROR createMember(
  TU* tu,                   /**< \ref TU environment. */
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
  data->markerOfParent = -1;
  data->markerToParent = -1;
  data->firstEdge = -1;
  data->representativeMember = -1;
  data->numEdges = 0;
  data->parentMember = -1;
  data->type = type;
  *pmember = tdec->numMembers;
  tdec->numMembers++;

  TUdbgMsg(10, "Creating %s member %d.\n",
    type == TDEC_MEMBER_TYPE_BOND ? "bond" : (type == TDEC_MEMBER_TYPE_PRIME ? "prime" : "polygon"), *pmember);

  return TU_OKAY;
}

TU_ERROR TUtdecCreate(TU* tu, TU_TDEC** ptdec, int memEdges, int memNodes, int memMembers,
  int numRows, int numColumns)
{
  assert(tu);
  assert(ptdec);
  assert(!*ptdec);

  TU_CALL( TUallocBlock(tu, ptdec) );
  TU_TDEC* tdec = *ptdec;
  if (memMembers < numRows)
    memMembers = numRows;
  tdec->memMembers = memMembers;
  tdec->numMembers = 0;
  tdec->members = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->members, tdec->memMembers) );

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

  if (memEdges < 1)
    memEdges = 1;
  tdec->memEdges = memEdges;
  tdec->edges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->edges, memEdges) );
  tdec->numEdges = 0;
  tdec->numMarkers = 0;

  /* Initialize free list with unused edges. */
  if (memEdges > tdec->numEdges)
  {
    for (int e = tdec->numEdges; e < memEdges; ++e)
    {
      tdec->edges[e].next = e+1;
      tdec->edges[e].member = -1;
    }
    tdec->edges[memEdges-1].next = -1;
    tdec->firstFreeEdge = tdec->numEdges;
  }
  else
    tdec->firstFreeEdge = -1;

  tdec->numRows = numRows;
  tdec->memRows = numRows;
  tdec->rowEdges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->rowEdges, tdec->numRows) );
  for (int r = 0; r < tdec->numRows; ++r)
    tdec->rowEdges[r].edge = -1;

  tdec->numColumns = numColumns;
  tdec->memColumns = tdec->numColumns;
  tdec->columnEdges = NULL;
  TU_CALL( TUallocBlockArray(tu, &tdec->columnEdges, tdec->numColumns) );
  for (int c = 0; c < tdec->numColumns; ++c)
    tdec->columnEdges[c].edge = -1;

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

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

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  TUdbgMsg(0, "TUtdecToGraph for t-decomposition.\n");

  TU_CALL( TUgraphClear(tu, graph) );

  TU_GRAPH_EDGE* localEdgeElements = NULL;
  if (edgeElements)
    localEdgeElements = edgeElements;
  else if (basis || cobasis)
    TU_CALL( TUallocStackArray(tu, &localEdgeElements, tdec->memEdges) );
  TU_GRAPH_NODE* tdecNodesToGraphNodes = NULL;
  TU_CALL( TUallocStackArray(tu, &tdecNodesToGraphNodes, tdec->memNodes) );
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
    TUdbgMsg(2, "Member %d is %s with %d edges.\n", member,
      type == TDEC_MEMBER_TYPE_BOND ? "a bond" : (type == TDEC_MEMBER_TYPE_POLYGON ? "a polygon" : "prime"),
       tdec->members[member].numEdges);

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
        TU_CALL( TUgraphAddEdge(tu, graph, graphHead, graphTail, &graphEdge) );
        tdecEdgesToGraphEdges[edge] = graphEdge;
        if (localEdgeElements)
          localEdgeElements[graphEdge] = tdec->edges[edge].name;
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
    TUdbgMsg(2, "Before merging, the graph has %d nodes and %d edges.\n", TUgraphNumNodes(graph),
      TUgraphNumEdges(graph));

    for (int m = 0; m < tdec->numMembers; ++m)
    {
      if (!isRepresentativeMember(tdec, m) || tdec->members[m].parentMember < 0)
        continue;

      TU_GRAPH_EDGE parent = tdecEdgesToGraphEdges[tdec->members[m].markerOfParent];
      TU_GRAPH_EDGE child = tdecEdgesToGraphEdges[tdec->members[m].markerToParent];
      TU_GRAPH_NODE parentU = TUgraphEdgeU(graph, parent);
      TU_GRAPH_NODE parentV = TUgraphEdgeV(graph, parent);
      TU_GRAPH_NODE childU = TUgraphEdgeU(graph, child);
      TU_GRAPH_NODE childV = TUgraphEdgeV(graph, child);

      TUdbgMsg(2, "Merging edges %d = {%d,%d} <%d> and %d = {%d,%d} <%d>.\n", parent, parentU, parentV,
        tdec->edges[tdec->members[m].markerOfParent].name, child, childU, childV,
        tdec->edges[tdec->members[m].markerToParent].name);

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
#if !defined(NDEBUG)
    /* This is only relevant if a 1-separation exists. */
    for (int r = 0; r < tdec->numRows; ++r)
      basis[r] = INT_MIN;
    for (int c = 0; c < tdec->numColumns; ++c)
      cobasis[c] = INT_MIN;
#endif /* !NDEBUG */

    for (TU_GRAPH_ITER i = TUgraphEdgesFirst(graph); TUgraphEdgesValid(graph, i);
      i = TUgraphEdgesNext(graph, i))
    {
      TU_GRAPH_EDGE e = TUgraphEdgesEdge(graph, i);

      TUdbgMsg(2, "Graph edge %d = {%d,%d} <%d>\n", e, TUgraphEdgeU(graph, e), TUgraphEdgeV(graph, e),
        localEdgeElements[e]);

      int element = localEdgeElements[e];
      if (element >= 0 && basis)
        basis[element] = e;
      else if (element < 0 && cobasis)
        cobasis[-1-element] = e;
    }

#if !defined(NDEBUG)
    /* These assertions indicate a 1-separable input matrix. */
    for (int r = 0; r < tdec->numRows; ++r)
      assert(basis[r] >= 0);
    for (int c = 0; c < tdec->numColumns; ++c)
      assert(cobasis[c] >= 0);
#endif /* !NDEBUG */
  }

  TU_CALL( TUfreeStackArray(tu, &tdecEdgesToGraphEdges) );
  TU_CALL( TUfreeStackArray(tu, &tdecNodesToGraphNodes) );
  if (localEdgeElements != edgeElements)
    TU_CALL( TUfreeStackArray(tu, &localEdgeElements) );

  return TU_OKAY;
}

static
void edgeToDot(
  FILE* stream,           /**< File stream. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_MEMBER member,  /**< Member this edge belongs to. */
  TU_TDEC_EDGE edge,      /**< Edge. */
  int u,                  /**< First node. */
  int v,                  /**< Second node. */
  bool red                /**< Whether to color it red. */
)
{
  assert(stream);
  assert(member >= 0);
  assert(edge >= 0);

  const char* redStyle = red ? ",color=red" : "";
  if (tdec->members[member].markerToParent == edge)
  {
    fprintf(stream, "    %d.%d -- p%d [label=\"%d\",style=dashed%s];\n", member, u, member, edge, redStyle);
    fprintf(stream, "    p%d -- %d.%d [label=\"%d\",style=dashed%s];\n", member, member, v, edge, redStyle);
    fprintf(stream, "    %d.%d [shape=box];\n", member, u);
    fprintf(stream, "    %d.%d [shape=box];\n", member, v);
    fprintf(stream, "    p%d [style=dashed];\n", member);
  }
  else if (tdec->edges[edge].childMember >= 0)
  {
    TU_TDEC_MEMBER child = tdec->edges[edge].childMember;
    fprintf(stream, "    %d.%d -- c%d [label=\"%d\",style=dotted%s];\n", member, u, child, edge, redStyle);
    fprintf(stream, "    c%d -- %d.%d [label=\"%d\",style=dotted%s];\n", child, member, v, edge,redStyle);
    fprintf(stream, "    %d.%d [shape=box];\n", member, u);
    fprintf(stream, "    %d.%d [shape=box];\n", member, v);
    fprintf(stream, "    c%d [style=dotted];\n", child);

    fprintf(stream, "    p%d -- c%d [style=dashed,dir=forward];\n", child, child);
  }
  else
  {
    fflush(stdout);
    fprintf(stream, "    %d.%d -- %d.%d [label=\"%d <%d>\",style=bold%s];\n", member, u, member, v,
      edge, tdec->edges[edge].name, redStyle);
    fprintf(stream, "    %d.%d [shape=box];\n", member, u);
    fprintf(stream, "    %d.%d [shape=box];\n", member, v);
  }
}

TU_ERROR TUtdecToDot(TU* tu, TU_TDEC* tdec, FILE* stream, bool* edgesHighlighted)
{
  assert(tu);
  assert(tdec);
  assert(stream);

  fprintf(stream, "// t-decomposition\n");
  fprintf(stream, "graph tdec {\n");
  fprintf(stream, "  compound = true;\n");
  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (!isRepresentativeMember(tdec, member))
      continue;

    fprintf(stream, "  subgraph member%d {\n", member);
    if (tdec->members[member].type == TDEC_MEMBER_TYPE_BOND)
    {
      TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
      do
      {
        edgeToDot(stream, tdec, member, edge, 0, 1, edgesHighlighted ? edgesHighlighted[edge] : false);
        edge = tdec->edges[edge].next;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    else if (tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME)
    {
      TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
      do
      {
        TU_TDEC_NODE u = findEdgeHead(tdec, edge);
        TU_TDEC_NODE v = findEdgeTail(tdec, edge);
        edgeToDot(stream, tdec, member, edge, u, v,
          edgesHighlighted ? edgesHighlighted[edge] : false);
        edge = tdec->edges[edge].next;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    else
    {
      assert(tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON);
      TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
      int i = 0;
      do
      {
        edgeToDot(stream, tdec, member, edge, i, (i+1) % tdec->members[member].numEdges,
          edgesHighlighted ? edgesHighlighted[edge] : false);
        edge = tdec->edges[edge].next;
        i++;
      }
      while (edge != tdec->members[member].firstEdge);
    }
    fprintf(stream, "  }\n");
  }
  fprintf(stream, "}\n");

  return TU_OKAY;
}

#if defined(TU_DEBUG_DOT)

static int dotFileCounter = 1;

static
TU_ERROR debugDot(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn  /**< new column. */
)
{
  char name[256];
  snprintf(name, 256, "tdec-%03d.dot", dotFileCounter);
  TUdbgMsg(0, "Writing <%s>...", name);
  FILE* dotFile = fopen(name, "w");
  TU_CALL( TUtdecToDot(tu, tdec, dotFile, newcolumn->edgesInPath) );
  fclose(dotFile);
  TUdbgMsg(0, " done.\n");

  dotFileCounter++;

  return TU_OKAY;
}

#endif /* TU_DEBUG_DOT */


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

  newcolumn->numReducedComponents = 0;
  newcolumn->memReducedComponents = 0;
  newcolumn->reducedComponents = NULL;

  newcolumn->pathEdgeStorage = NULL;
  newcolumn->memPathEdges = 0;
  newcolumn->numPathEdges = 0;

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
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->edgesInPath) );

  if (newcolumn->nodesDegree)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->nodesDegree) );

  if (newcolumn->reducedComponents)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->reducedComponents) );
  if (newcolumn->reducedMembers)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->reducedMembers) );

  if (newcolumn->membersToReducedMembers)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->membersToReducedMembers) );
  if (newcolumn->pathEdgeStorage)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->pathEdgeStorage) );
  if (newcolumn->childrenStorage)
    TU_CALL( TUfreeBlockArray(tu, &newcolumn->childrenStorage) );

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
  assert(newcolumn->numReducedMembers == 0);

  newcolumn->remainsGraphic = true;
  newcolumn->numPathEdges = 0;

  // TODO: Remember sizes of these arrays.

  /* memEdges does not suffice since new edges can be created by squeezing off.
   * Each squeezing off introduces 4 new edges, and we might apply this twice for each polygon member. */
  TU_CALL( TUreallocBlockArray(tu, &newcolumn->edgesInPath, tdec->memEdges + 8*tdec->numMembers + 32) );
  TU_CALL( TUreallocBlockArray(tu, &newcolumn->nodesDegree, tdec->memNodes) );

#if defined(TU_DEBUG_DOT)
  for (int e = 0; e < tdec->memEdges + 8*tdec->numMembers + 32; ++e)
    newcolumn->edgesInPath[e] = false;
#endif /* TU_DEBUG_DOT */

  return TU_OKAY;
}

/**
 * \brief Creates, if necessary, the reduced member for \p member and calls itself for the parent.
 */

static
ReducedMember* createReducedMember(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new column. */
  TU_TDEC_MEMBER member,              /**< Member to create reduced member for. */
  ReducedMember** rootDepthMinimizer  /**< Array mapping root members to the depth minimizer. */
)
{
#if defined(TU_DEBUG_TDEC)
  printf("        Attempting to create reduced member %d.\n", member);
#endif /* TU_DEBUG_TDEC */

  ReducedMember* reducedMember = newcolumn->membersToReducedMembers[member];
  if (reducedMember)
  {
    /* This member is a known reduced member. If we meet an existing path of low depth, we remember
     * that. */

    TUdbgMsg(10, "Reduced member exists.\n");

    if (!rootDepthMinimizer[reducedMember->rootMember] ||
      reducedMember->depth < rootDepthMinimizer[reducedMember->rootMember]->depth)
    {
      TUdbgMsg(8, "Updating depth to %d.\n", reducedMember->depth);
      rootDepthMinimizer[reducedMember->rootMember] = reducedMember;
    }
  }
  else
  {
    reducedMember = &newcolumn->reducedMembers[newcolumn->numReducedMembers];
    newcolumn->numReducedMembers++;
    newcolumn->membersToReducedMembers[member] = reducedMember;
    reducedMember->member = member;
    reducedMember->numChildren = 0;
    reducedMember->polygonPathEdge = -1;
    reducedMember->polygonNonpathEdge = -1;
    reducedMember->primeEndNodes[0] = -1;
    reducedMember->primeEndNodes[1] = -1;
    reducedMember->primeEndNodes[2] = -1;
    reducedMember->primeEndNodes[3] = -1;

    TUdbgMsg(10, "Reduced member is new.\n");

    TU_TDEC_MEMBER parentMember = findMemberParent(tdec, member);
    if (parentMember >= 0)
    {
      ReducedMember* parentReducedMember = createReducedMember(tu, tdec, newcolumn, parentMember,
        rootDepthMinimizer);
      reducedMember->parent = parentReducedMember;
      reducedMember->depth = parentReducedMember->depth + 1;
      reducedMember->rootMember = parentReducedMember->rootMember;
      parentReducedMember->numChildren++;
    }
    else
    {
      reducedMember->parent = NULL;
      reducedMember->depth = 0;
      reducedMember->rootMember = member;

      /* We found a new component. We temporarily store the root member and later compute the actual
         reduced root. */
      if (newcolumn->memReducedComponents == newcolumn->numReducedComponents)
      {
        newcolumn->memReducedComponents = 2 * newcolumn->memReducedComponents + 16;
        TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedComponents,
          newcolumn->memReducedComponents) );
      }
      TUdbgMsg(8, "Initializing the new reduced component %d.\n", newcolumn->numReducedComponents);

      newcolumn->reducedComponents[newcolumn->numReducedComponents].root = reducedMember;
      newcolumn->numReducedComponents++;
    }

    TUdbgMsg(8, "The root member of %d is %d.\n", reducedMember->member, reducedMember->rootMember);
  }
  return reducedMember;
}

static
TU_ERROR computeReducedDecomposition(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  int* entryRows,               /**< Array of rows of new column's enries. */
  int numEntries                /**< Length of \p entryRows. */
)
{
  /* Identify all members on the path. For the induced sub-arborescence we also compute the
   * depths. After the computation, its root has depth pathRootDepth. */
  TUdbgMsg(4, "Computing reduced t-decomposition.\n");

  /* Enlarge members array. */
  int maxRow = 0;
  for (int p = 0; p < numEntries; ++p)
  {
    if (entryRows[p] > maxRow)
      maxRow = entryRows[p];
  }
  if (newcolumn->memReducedMembers < tdec->numMembers + numEntries)
  {
    newcolumn->memReducedMembers = tdec->memMembers + maxRow + 1;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedMembers, newcolumn->memReducedMembers) );
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->membersToReducedMembers,
      newcolumn->memReducedMembers) );
  }

  /* Initialize the mapping from members to reduced members. */
  for (int m = 0; m < tdec->numMembers; ++m)
    newcolumn->membersToReducedMembers[m] = NULL;

  ReducedMember** rootDepthMinimizer = NULL;
  TU_CALL( TUallocStackArray(tu, &rootDepthMinimizer, tdec->numMembers) );
  for (int m = 0; m < tdec->numMembers; ++m)
    rootDepthMinimizer[m] = NULL;
  newcolumn->numReducedMembers = 0;
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    TU_TDEC_EDGE edge = (row < tdec->numRows) ? tdec->rowEdges[row].edge : -1;
    TUdbgMsg(6, "Entry %d is row %d of %d and corresponds to edge %d.\n", p, row, tdec->numRows, edge);
    if (edge >= 0)
    {
      TU_TDEC_MEMBER member = findEdgeMember(tdec, edge);
      TUdbgMsg(8, "Edge %d exists and belongs to member %d.\n", edge, member);
      ReducedMember* reducedMember = createReducedMember(tu, tdec, newcolumn, member, rootDepthMinimizer);

      /* For the first edge of this member, we set the depth minimizer to the new reduced member. */
      if (!rootDepthMinimizer[reducedMember->rootMember])
      {
        rootDepthMinimizer[reducedMember->rootMember] = reducedMember;
      }
    }
  }

  /* We set the reduced roots according to the minimizers. */
  for (int i = 0; i < newcolumn->numReducedComponents; ++i)
  {
    TUdbgMsg(6, "Considering reduced component %d with initial root member %d.\n", i,
      newcolumn->reducedComponents[i].root->member);
    TUdbgMsg(8, "The minimizer is %d.\n",
      rootDepthMinimizer[newcolumn->reducedComponents[i].root->member]->member);

    newcolumn->reducedComponents[i].rootDepth =
      rootDepthMinimizer[newcolumn->reducedComponents[i].root->member]->depth;
    newcolumn->reducedComponents[i].root =
      rootDepthMinimizer[newcolumn->reducedComponents[i].root->member];
    newcolumn->reducedComponents[i].numTerminals = 0;
    TUdbgMsg(8, "Member %d is the new root of the reduced decomposition of this component.\n",
      newcolumn->reducedComponents[i].root->member);
  }

  /* Allocate memory for children. */
  if (newcolumn->memChildrenStorage < newcolumn->numReducedMembers)
  {
    newcolumn->memChildrenStorage = 2*newcolumn->numReducedMembers;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->childrenStorage, newcolumn->memChildrenStorage) );
  }
  newcolumn->usedChildrenStorage = 0;

  /* Set memory pointer of each reduced member. */
  for (int m = 0; m < newcolumn->numReducedMembers; ++m)
  {
    ReducedMember* reducedMember = &newcolumn->reducedMembers[m];
    if (reducedMember->depth >= rootDepthMinimizer[reducedMember->rootMember]->depth)
    {
      TUdbgMsg(6, "Member %d's depth is greater than that of the root, and it has %d children.\n",
        reducedMember->member, reducedMember->numChildren);
      reducedMember->children = &newcolumn->childrenStorage[newcolumn->usedChildrenStorage];
      newcolumn->usedChildrenStorage += reducedMember->numChildren;
      reducedMember->numChildren = 0;
    }
    else
    {
      TUdbgMsg(6, "Member %d's depth is smaller than that of its new root.\n", reducedMember->member);
      continue;
    }
  }

  TUdbgMsg(4, "Total number of children is %d / %d.\n", newcolumn->usedChildrenStorage, newcolumn->memChildrenStorage);

  /* Set children of each reduced member. */
  for (int m = 0; m < newcolumn->numReducedMembers; ++m)
  {
    ReducedMember* reducedMember = &newcolumn->reducedMembers[m];
    if (reducedMember->depth <= rootDepthMinimizer[reducedMember->rootMember]->depth)
    {
      TUdbgMsg(6, "Member %d's depth is smaller than or equal to that of its reduced root.\n", reducedMember->member);
      continue;
    }

    TU_TDEC_MEMBER parentMember = findMemberParent(tdec, newcolumn->reducedMembers[m].member);
    ReducedMember* parentReducedMember = parentMember >= 0 ? newcolumn->membersToReducedMembers[parentMember] : NULL;
    TUdbgMsg(6, "Member %d's depth is greater than that of its reduced root. Its parent is %d, and reduced parent %p.\n",
      reducedMember->member, parentMember, parentReducedMember);

    if (parentReducedMember)
    {
      TUdbgMsg(6, "Reduced member %ld (= member %d) has %d (= member %d) as child %d.\n",
        (parentReducedMember - newcolumn->reducedMembers),
        parentReducedMember->member, m, newcolumn->reducedMembers[m].member, parentReducedMember->numChildren);
      parentReducedMember->children[parentReducedMember->numChildren] = &newcolumn->reducedMembers[m];
      parentReducedMember->numChildren++;
    }
  }

  TU_CALL( TUfreeStackArray(tu, &rootDepthMinimizer) );

  return TU_OKAY;
}

/**
 * \brief Creates members and reduced members of new edges.
 */

static
TU_ERROR completeReducedDecomposition(
  TU* tu,                       /**< \ref TU environment. */
  TU_TDEC* tdec,                /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn, /**< new column. */
  int* entryRows,               /**< Array of rows of new column's enries. */
  int numEntries                /**< Length of \p entryRows. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);

  /* Check if we need new rows. */

  int newNumRows = tdec->numRows-1;
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    TU_TDEC_EDGE edge = (row < tdec->numRows) ? tdec->rowEdges[row].edge : -1;
    if (edge < 0)
    {
      if (row > newNumRows)
        newNumRows = row;
    }
  }
  newNumRows++;

  TUdbgMsg(4, "Completing reduced decomposition: increasing #rows from %d to %d.\n", tdec->numRows, newNumRows);

  /* Create single-edge bond members for all new rows. */

  if (newNumRows > tdec->numRows)
  {
    if (newNumRows > tdec->memRows)
    {
      tdec->memRows = 2*newNumRows;
      TU_CALL( TUreallocBlockArray(tu, &tdec->rowEdges, tdec->memRows) );
    }

    for (int r = tdec->numRows; r < newNumRows; ++r)
    {
      TU_TDEC_MEMBER member;
      TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_BOND, &member) );
      
      TU_TDEC_EDGE edge;
      TU_CALL( createEdge(tu, tdec, member, &edge) );
      TU_CALL( addEdgeToMembersEdgeList(tu, tdec, edge, member) );
      tdec->edges[edge].name = r;
      tdec->edges[edge].head = -1;
      tdec->edges[edge].tail = -1;
      tdec->edges[edge].childMember = -1;

      TUdbgMsg(8, "New row %d is edge %d of member %d.\n", r, edge, member);

      tdec->rowEdges[r].edge = edge;
    }
  }

  /* Create reduced members. */
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    if (row >= tdec->numRows)
    {
      /* Edge and member for this row were created. We create the reduced member. */
      TU_TDEC_EDGE edge = tdec->rowEdges[row].edge;
      TU_TDEC_MEMBER member = findEdgeMember(tdec, edge);

      TUdbgMsg(4, "Creating reduced member for edge %d of member %d.\n", edge, member);

      assert(newcolumn->numReducedMembers < newcolumn->memReducedMembers);
      ReducedMember* reducedMember = &newcolumn->reducedMembers[newcolumn->numReducedMembers];
      newcolumn->numReducedMembers++;
      reducedMember->numChildren = 0;
      reducedMember->member = member;
      reducedMember->depth = 0;
      reducedMember->rootMember = -1;
      reducedMember->type = TYPE_5_ROOT;

      assert(newcolumn->numPathEdges + 1 < newcolumn->memPathEdges);
      PathEdge* reducedEdge = &newcolumn->pathEdgeStorage[newcolumn->numPathEdges];
      newcolumn->numPathEdges++;
      reducedEdge->next = NULL;
      reducedEdge->edge = edge;
      reducedMember->firstPathEdge = reducedEdge;

      if (newcolumn->numReducedComponents == newcolumn->memReducedComponents)
      {
        newcolumn->memReducedComponents = 2 * newcolumn->memReducedComponents + 16;
        TU_CALL( TUreallocBlockArray(tu, &newcolumn->reducedComponents,
          newcolumn->memReducedComponents) );
      }

      newcolumn->membersToReducedMembers[member] = reducedMember;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].root = reducedMember;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].rootDepth = 0;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].numTerminals = 0;
#if !defined(NDEBUG)
      newcolumn->reducedComponents[newcolumn->numReducedComponents].terminalMember[0] = INT_MIN;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].terminalMember[1] = INT_MIN;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].terminalNode[0] = INT_MIN;
      newcolumn->reducedComponents[newcolumn->numReducedComponents].terminalNode[1] = INT_MIN;
#endif /* !NDEBUG */
      newcolumn->numReducedComponents++;
    }
  }

  tdec->numRows = newNumRows;

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
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  
  TUdbgMsg(4, "Initializing edge lists for members of reduced t-decomposition.\n");

  for (int v = 0; v < tdec->memNodes; ++v)
    newcolumn->nodesDegree[v] = 0;
  for (int e = 0; e < tdec->memEdges; ++e)
    newcolumn->edgesInPath[e] = false;

  /* (Re)allocate memory for edge lists. */
  assert(newcolumn->numPathEdges == 0);
  int requiredMemPathEdges = numEntries + newcolumn->numReducedMembers;
  if (newcolumn->memPathEdges < requiredMemPathEdges)
  {
    newcolumn->memPathEdges = 2 * requiredMemPathEdges;
    TU_CALL( TUreallocBlockArray(tu, &newcolumn->pathEdgeStorage,
      newcolumn->memPathEdges) );
  }

  /* Start with empty lists. */
  for (int i = 0; i < newcolumn->numReducedMembers; ++i)
    newcolumn->reducedMembers[i].firstPathEdge = NULL;

  /* Fill edge lists. */
  for (int p = 0; p < numEntries; ++p)
  {
    int row = entryRows[p];
    TU_TDEC_EDGE edge = (row < tdec->numRows) ? tdec->rowEdges[row].edge : -1;
    if (edge >= 0)
    {
      TU_TDEC_MEMBER member = findEdgeMember(tdec, edge);
      assert(member >= 0);
      ReducedMember* reducedMember = newcolumn->membersToReducedMembers[member];
      assert(reducedMember);
      newcolumn->pathEdgeStorage[newcolumn->numPathEdges].next = reducedMember->firstPathEdge;
      newcolumn->pathEdgeStorage[newcolumn->numPathEdges].edge = edge;
      reducedMember->firstPathEdge = &newcolumn->pathEdgeStorage[newcolumn->numPathEdges];
      ++newcolumn->numPathEdges;

      newcolumn->edgesInPath[edge] = true;
      if (tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME)
      {
        newcolumn->nodesDegree[findEdgeHead(tdec, edge)]++;
        newcolumn->nodesDegree[findEdgeTail(tdec, edge)]++;
      }

      TUdbgMsg(6, "Edge %d <%d> belongs to reduced member %ld which is member %d.\n", edge, tdec->edges[edge].name,
        (reducedMember - newcolumn->reducedMembers), reducedMember->member);
    }
  }

  return TU_OKAY;
}

/**
 * \brief Count the number of children of a reduced member having certain types.
 */

static
TU_ERROR countChildrenTypes(
  TU* tu,                           /**< \ref TU environment. */
  TU_TDEC* tdec,                    /**< t-decomposition. */
  ReducedMember* reducedMember,     /**< Reduced member. */
  int* pNumOneEnd,                  /**< Number of children that (recursively) must contain one path end. */
  int* pNumTwoEnds,                 /**< Number of children that (recursively) must contain two path ends. */
  TU_TDEC_EDGE childMarkerEdges[2]  /**< Array for storing a child marker edges containing one/two path ends. */
)
{
  assert(tu);
  assert(tdec);
  assert(reducedMember);
  assert(pNumOneEnd);
  assert(pNumTwoEnds);
  assert(childMarkerEdges);

  *pNumOneEnd = 0;
  *pNumTwoEnds = 0;
  childMarkerEdges[0] = -1;
  childMarkerEdges[1] = -1;
  int nextChildMarker = 0;

  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    ReducedMember* child = reducedMember->children[c];
    assert(child);

    if (child->type == TYPE_2_SHORTCUT || child->type == TYPE_3_EXTENSION)
    {
      if (nextChildMarker < 2)
      {
        childMarkerEdges[nextChildMarker] = tdec->members[findMember(tdec, child->member)].markerOfParent;
        nextChildMarker++;
      }
      (*pNumOneEnd)++;
    }
    else if (child->type == TYPE_4_CONNECTS_TWO_PATHS)
    {
      if (nextChildMarker < 2)
      {
        childMarkerEdges[nextChildMarker] = tdec->members[findMember(tdec, child->member)].markerOfParent;
        nextChildMarker++;
      }
      (*pNumTwoEnds)++;
    }
  }

  return TU_OKAY;
}

static
TU_ERROR addTerminal(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  TU_TDEC_MEMBER member,              /**< New terminal member. */
  TU_TDEC_NODE node                   /**< New terminal node. */
)
{
  assert(reducedComponent);
  assert(member >= 0);
  assert(isRepresentativeMember(tdec, member));

  /* For bonds we don't need to provide a node. */
  assert(node >= 0 || tdec->members[member].type == TDEC_MEMBER_TYPE_BOND);
  assert(reducedComponent->numTerminals != 1 || node >= 0 || member == reducedComponent->terminalMember[0]);

  if (reducedComponent->numTerminals == 2)
  {
    TUdbgMsg(8, "Attempted to add terminal but already 2 known. We should detect non-graphicness soon.\n");
  }
  else
  {
    reducedComponent->terminalMember[reducedComponent->numTerminals] = member;
    reducedComponent->terminalNode[reducedComponent->numTerminals] = node;
    reducedComponent->numTerminals++;
  }

  return TU_OKAY;
}

static
TU_ERROR determineTypeBond(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int numOneEnd,                      /**< Number of child markers containing one path end. */
  int numTwoEnds,                     /**< Number of child markers containing two path ends. */
  TU_TDEC_EDGE childMarkerEdges[2],   /**< Marker edges of children containing one/two path ends. */
  int depth                           /**< Depth of member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedComponent);
  assert(reducedMember);
  assert(numOneEnd >= 0);
  assert(numTwoEnds >= 0);
  assert(numOneEnd + 2*numTwoEnds <= 2);
  assert(childMarkerEdges);
  assert(tdec->members[findMember(tdec, reducedMember->member)].type == TDEC_MEMBER_TYPE_BOND);

  if (depth == 0)
  {
    /* A bond root always works. */
    reducedMember->type = TYPE_5_ROOT;
    return TU_OKAY;
  }

  /* No children, but a reduced edge. */
  if (2*numTwoEnds + numOneEnd == 0 && reducedMember->firstPathEdge)
    reducedMember->type = TYPE_1_CLOSES_CYCLE;
  else if (numOneEnd == 1)
    reducedMember->type = reducedMember->firstPathEdge ? TYPE_2_SHORTCUT : TYPE_3_EXTENSION;
  else if (numOneEnd + 2 * numTwoEnds == 2)
  {
    if (reducedMember->firstPathEdge)
    {
      /* If there is a path edge then this should have been the root, but it is not! */
      newcolumn->remainsGraphic = false;
    }
    else
      reducedMember->type = TYPE_4_CONNECTS_TWO_PATHS;
  }
  else
  {
    /* Since no children contains path edges, the bond must be a leaf of the reduced decomposition and contains one. */
    assert(reducedMember->firstPathEdge);
    reducedMember->type = TYPE_1_CLOSES_CYCLE;
  }

  return TU_OKAY;
}

static
TU_ERROR determineTypePolygon(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int numOneEnd,                      /**< Number of child markers containing one path end. */
  int numTwoEnds,                     /**< Number of child markers containing two path ends. */
  TU_TDEC_EDGE childMarkerEdges[2],   /**< Marker edges of children containing one/two path ends. */
  int depth                           /**< Depth of member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedComponent);
  assert(reducedMember);
  assert(numOneEnd >= 0);
  assert(numTwoEnds >= 0);
  assert(numOneEnd + 2*numTwoEnds <= 2);
  assert(childMarkerEdges);
  assert(tdec->members[findMember(tdec, reducedMember->member)].type == TDEC_MEMBER_TYPE_POLYGON);

  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);

  if (depth == 0)
  {
    /* We assume that we are not the root of the whole decomposition. */
    assert(tdec->members[member].parentMember >= 0);

    newcolumn->remainsGraphic = numTwoEnds == 0;
    reducedMember->type = TYPE_5_ROOT;
    return TU_OKAY;
  }

  int countReducedEdges = 0;
  for (PathEdge* edge = reducedMember->firstPathEdge; edge != NULL; edge = edge->next)
    ++countReducedEdges;
  int numEdges = tdec->members[member].numEdges;
  if (countReducedEdges == numEdges - 1)
  {
    reducedMember->type = TYPE_1_CLOSES_CYCLE;
  }
  else if (countReducedEdges + numTwoEnds == numEdges - 1)
  {
    assert(numTwoEnds == 1);
    reducedMember->type = TYPE_4_CONNECTS_TWO_PATHS;
  }
  else if (numTwoEnds == 1)
  {
    newcolumn->remainsGraphic = false;
  }
  else if (numOneEnd == 1)
  {
    reducedMember->type = TYPE_3_EXTENSION;
  }
  else if (numOneEnd == 2)
  {
    reducedMember->type = TYPE_4_CONNECTS_TWO_PATHS;
  }
  else
  {
    assert(numOneEnd == 0);
    assert(numTwoEnds == 0);
    reducedMember->type = TYPE_3_EXTENSION;
  }

  return TU_OKAY;
}

static
TU_ERROR determineTypePrime(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int numOneEnd,                      /**< Number of child markers containing one path end. */
  int numTwoEnds,                     /**< Number of child markers containing two path ends. */
  TU_TDEC_EDGE childMarkerEdges[2],   /**< Marker edges of children containing one/two path ends. */
  int depth                           /**< Depth of member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedComponent);
  assert(reducedMember);
  assert(numOneEnd >= 0);
  assert(numTwoEnds >= 0);
  assert(numOneEnd + 2*numTwoEnds <= 2);
  assert(childMarkerEdges);
  assert(tdec->members[findMember(tdec, reducedMember->member)].type == TDEC_MEMBER_TYPE_PRIME);

  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);

  /* Collect nodes of parent marker and of child markers containing one/two path end nodes. */
  TU_TDEC_NODE parentMarkerNodes[2] = {
    depth == 0 ? -1 : findEdgeTail(tdec, tdec->members[member].markerToParent),
    depth == 0 ? -1 : findEdgeHead(tdec, tdec->members[member].markerToParent)
  };
  TU_TDEC_NODE childMarkerNodes[4] = {
    childMarkerEdges[0] < 0 ? -1 : findEdgeTail(tdec, childMarkerEdges[0]),
    childMarkerEdges[0] < 0 ? -1 : findEdgeHead(tdec, childMarkerEdges[0]),
    childMarkerEdges[1] < 0 ? -1 : findEdgeTail(tdec, childMarkerEdges[1]),
    childMarkerEdges[1] < 0 ? -1 : findEdgeHead(tdec, childMarkerEdges[1])
  };

  /* Check the node degrees (with respect to path edges) in this component. */
  int numEndNodes = 0;
  for (PathEdge* reducedEdge = reducedMember->firstPathEdge; reducedEdge; reducedEdge = reducedEdge->next)
  {
    TU_TDEC_NODE nodes[2] = { findEdgeHead(tdec, reducedEdge->edge), findEdgeTail(tdec, reducedEdge->edge) };
    for (int i = 0; i < 2; ++i)
    {
      TU_TDEC_NODE v = nodes[i];
      if (newcolumn->nodesDegree[v] >= 3)
      {
        TUdbgMsg(6 + 2*depth, "Node %d of prime member %d has path-degree at least 3.\n", v, reducedMember->member);
        newcolumn->remainsGraphic = false;
        return TU_OKAY;
      }

      if (newcolumn->nodesDegree[v] == 1)
      {
        if (numEndNodes == 4)
        {
          TUdbgMsg(6 + 2*depth, "Prime member %d has at least five path end nodes: %d, %d, %d, %d and %d.\n",
            reducedMember->member, reducedMember->primeEndNodes[0], reducedMember->primeEndNodes[1],
            reducedMember->primeEndNodes[2], reducedMember->primeEndNodes[3], v);
          newcolumn->remainsGraphic = false;
          return TU_OKAY;
        }

        reducedMember->primeEndNodes[numEndNodes] = v;
        ++numEndNodes;
      }
    }
  }

  TUdbgMsg(6 + 2*depth, "Prime member %d has %d path end nodes", member, numEndNodes);
  for (int i = 0; i < numEndNodes; ++i)
    TUdbgMsg(0, ", %d", reducedMember->primeEndNodes[i]);
  TUdbgMsg(0, ".\n");

  if (depth == 0)
  {
    if (numEndNodes == 0)
    {
      assert(0 == "Typing of prime not fully implemented: non-root with 0 path nodes.");
    }
    else if (numEndNodes == 2)
    {
      if (numOneEnd == 0 && numTwoEnds == 0)
      {
        reducedMember->type = TYPE_5_ROOT;
        TU_CALL( addTerminal(tu, tdec, reducedComponent, member, reducedMember->primeEndNodes[0]) );
        TU_CALL( addTerminal(tu, tdec, reducedComponent, member, reducedMember->primeEndNodes[1]) );
      }
      else
      {
        assert(0 == "Typing of prime not fully implemented: non-root with 2 path nodes.");
      }
    }
    else if (numEndNodes == 4)
      newcolumn->remainsGraphic = false;
  }
  else
  {
    /* Non-root member. */
    int parentMarkerDegrees[2] = {
      newcolumn->nodesDegree[parentMarkerNodes[0]],
      newcolumn->nodesDegree[parentMarkerNodes[1]]
    };

    if (numEndNodes == 0)
    {
      /* We have no path edges, so there must be at least one child containing one/two path ends. */
      assert(numOneEnd + numTwoEnds > 0);
      /* We should not have a child marker edge parallel to the parent marker edge! */
      assert(!(parentMarkerNodes[0] == childMarkerNodes[0] && parentMarkerNodes[1] == childMarkerNodes[1])
        && !(parentMarkerNodes[0] == childMarkerNodes[1] && parentMarkerNodes[1] == childMarkerNodes[0]));

      if (numOneEnd == 0)
      {
        /* Even if parent and child marker (type 4) are adjacent, this is non-graphic. */
        newcolumn->remainsGraphic = false;
      }
      else if (numOneEnd == 1)
      {
        if (childMarkerNodes[0] == parentMarkerNodes[0] || childMarkerNodes[0] == parentMarkerNodes[1]
          || childMarkerNodes[1] == parentMarkerNodes[0] || childMarkerNodes[1] == parentMarkerNodes[1])
        {
          reducedMember->type = TYPE_3_EXTENSION;
        }
        else
        {
          newcolumn->remainsGraphic = false;
        }
      }
      else
      {
        assert(0 == "Typing of prime not fully implemented: non-root with 0 path nodes and 2 child markers with a path end.");
      }
    }
    else if (numEndNodes == 2)
    {
      /* Exchange such that end node 0 is at the parent marker. */
      if (reducedMember->primeEndNodes[0] != parentMarkerNodes[0] && reducedMember->primeEndNodes[0] != parentMarkerNodes[1])
      {
        TU_TDEC_NODE tmp = reducedMember->primeEndNodes[0];
        reducedMember->primeEndNodes[0] = reducedMember->primeEndNodes[1];
        reducedMember->primeEndNodes[1] = tmp;
      }
      assert(reducedMember->primeEndNodes[0] == parentMarkerNodes[0] || reducedMember->primeEndNodes[0] == parentMarkerNodes[1]);

      if (numOneEnd == 0 && numTwoEnds == 0)
      {
        if (parentMarkerDegrees[0] % 2 == 0 && parentMarkerDegrees[1] == 1)
        {
          reducedMember->type = parentMarkerDegrees[0] == 0 ? TYPE_3_EXTENSION : TYPE_2_SHORTCUT;
          TU_CALL( addTerminal(tu, tdec, reducedComponent, member, reducedMember->primeEndNodes[1]) );
        }
        else if (parentMarkerDegrees[0] == 1 && parentMarkerDegrees[1] % 2 == 0)
        {
          reducedMember->type = parentMarkerDegrees[1] == 0 ? TYPE_3_EXTENSION : TYPE_2_SHORTCUT;
          TU_CALL( addTerminal(tu, tdec, reducedComponent, member, reducedMember->primeEndNodes[1]) );
        }
        else if (parentMarkerDegrees[0] == 1 && parentMarkerDegrees[1] == 1)
        {
          reducedMember->type = TYPE_1_CLOSES_CYCLE;
        }
        else
        {
          /* Both have degree 0 or 2. */
          newcolumn->remainsGraphic = false;
        }
      }
      else
      {
        assert(0 == "Typing of prime not fully implemented: non-root with 2 path nodes.");
      }
    }
    else if (numEndNodes == 4)
    {
      /* We figure out which end nodes are connected. */
      
      TU_TDEC_EDGE* nodeEdges = NULL;
      TUallocStackArray(tu, &nodeEdges, 2*tdec->memNodes);
      
      /* Initialize relevant entries to -1. */
      for (PathEdge* reducedEdge = reducedMember->firstPathEdge; reducedEdge; reducedEdge = reducedEdge->next)
      {
        TU_TDEC_NODE nodes[2] = { findEdgeHead(tdec, reducedEdge->edge), findEdgeTail(tdec, reducedEdge->edge) };
        for (int i = 0; i < 2; ++i)
        {
          TU_TDEC_NODE v = nodes[i];
          nodeEdges[2*v] = -1;
          nodeEdges[2*v + 1] = -1;
        }
      }

      /* Store incident edges for every node. */
      for (PathEdge* reducedEdge = reducedMember->firstPathEdge; reducedEdge; reducedEdge = reducedEdge->next)
      {
        TU_TDEC_NODE nodes[2] = { findEdgeHead(tdec, reducedEdge->edge), findEdgeTail(tdec, reducedEdge->edge) };
        for (int i = 0; i < 2; ++i)
        {
          TU_TDEC_NODE v = nodes[i];
          nodeEdges[2*v + (nodeEdges[2*v] == -1 ? 0 : 1)] = reducedEdge->edge;
        }
      }

      /* Start at end node 0 and see where we end. */
      TU_TDEC_EDGE previousEdge = -1;
      TU_TDEC_NODE currentNode = reducedMember->primeEndNodes[0];
      while (true)
      {
        TU_TDEC_EDGE edge = nodeEdges[2*currentNode];
        if (edge == previousEdge)
          edge = nodeEdges[2*currentNode+1];
        if (edge == -1)
          break;
        previousEdge = edge;
        TU_TDEC_NODE v = findEdgeHead(tdec, edge);
        currentNode = (v != currentNode) ? v : findEdgeTail(tdec, edge);
      }
      TUfreeStackArray(tu, &nodeEdges);

      /* Exchange such that we end nodes 0 and 1 are end nodes of the same path. */
      if (currentNode == reducedMember->primeEndNodes[2])
      {
        reducedMember->primeEndNodes[2] = reducedMember->primeEndNodes[1];
        reducedMember->primeEndNodes[1] = currentNode;
      }
      else if (currentNode == reducedMember->primeEndNodes[3])
      {
        reducedMember->primeEndNodes[3] = reducedMember->primeEndNodes[1];
        reducedMember->primeEndNodes[1] = currentNode;
      }

      /* Exchange such that end node 0 is at the parent marker. */
      if (reducedMember->primeEndNodes[0] != parentMarkerNodes[0] && reducedMember->primeEndNodes[0] != parentMarkerNodes[1])
      {
        TU_TDEC_NODE tmp = reducedMember->primeEndNodes[0];
        reducedMember->primeEndNodes[0] = reducedMember->primeEndNodes[1];
        reducedMember->primeEndNodes[1] = tmp;
      }
      assert(reducedMember->primeEndNodes[0] == parentMarkerNodes[0] || reducedMember->primeEndNodes[0] == parentMarkerNodes[1]);

      /* Exchange such that end node 2 is at the parent marker. */
      if (reducedMember->primeEndNodes[2] != parentMarkerNodes[0] && reducedMember->primeEndNodes[2] != parentMarkerNodes[1])
      {
        TU_TDEC_NODE tmp = reducedMember->primeEndNodes[2];
        reducedMember->primeEndNodes[2] = reducedMember->primeEndNodes[3];
        reducedMember->primeEndNodes[3] = tmp;
      }
      assert(reducedMember->primeEndNodes[2] == parentMarkerNodes[0] || reducedMember->primeEndNodes[2] == parentMarkerNodes[1]);

      assert(0 == "Typing of prime not fully implemented: non-root with 4 path nodes.");
    }
  }

  return TU_OKAY;
}

static
TU_ERROR determineTypes(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int depth                           /**< Depth of member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);

  TUdbgMsg(4 + 2*depth, "determineTypes(member %d = reduced member %ld)\n", reducedMember->member,
    reducedMember - &newcolumn->reducedMembers[0]);

  /* First handle children recursively. */
  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    TU_CALL( determineTypes(tu, tdec, newcolumn, reducedComponent, reducedMember->children[c], depth + 1) );

    if (newcolumn->remainsGraphic)
    {
      TUdbgMsg(6 + 2*depth, "Child member %d of %d has type %d\n", reducedMember->children[c]->member, reducedMember->member,
        reducedMember->children[c]->type);
    }
    else
    {
      TUdbgMsg(6 + 2*depth, "Child prohibits graphicness.\n");
    }

    /* Abort if some part indicates non-graphicness. */
    if (!newcolumn->remainsGraphic)
      return TU_OKAY;
  }

  int numOneEnd, numTwoEnds;
  TU_TDEC_EDGE childMarkerEdges[2];
  TU_CALL( countChildrenTypes(tu, tdec, reducedMember, &numOneEnd, &numTwoEnds, childMarkerEdges) );

#if defined(TU_DEBUG_TDEC)
  int countReducedEdges = 0;
  for (PathEdge* e = reducedMember->firstPathEdge; e; e = e->next)
    ++countReducedEdges;
  TUdbgMsg(6 + 2*depth, "Member %d has %d children with one end and %d with two ends and %d reduced edges.\n",
    reducedMember->member, numOneEnd, numTwoEnds, countReducedEdges);
#endif /* TU_DEBUG_TDEC */

  if (2*numTwoEnds + numOneEnd > 2)
  {
    newcolumn->remainsGraphic = false;
    return TU_OKAY;
  }

  bool isRoot = reducedMember == reducedComponent->root;
  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);
  if (tdec->members[member].type == TDEC_MEMBER_TYPE_BOND)
  {
    TU_CALL( determineTypeBond(tu, tdec, newcolumn, reducedComponent, reducedMember, numOneEnd, numTwoEnds,
      childMarkerEdges, depth) );
  }
  else if (tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON)
  {
    TU_CALL( determineTypePolygon(tu, tdec, newcolumn, reducedComponent, reducedMember, numOneEnd, numTwoEnds,
      childMarkerEdges, depth) );
  }
  else
  {
    assert(tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME);
    TU_CALL( determineTypePrime(tu, tdec, newcolumn, reducedComponent, reducedMember, numOneEnd, numTwoEnds,
      childMarkerEdges, depth) );
  }

  /* Parent marker edge closes cycle, so we propagate information to parent. */

  if (!isRoot && reducedMember->type == TYPE_1_CLOSES_CYCLE)
  {
    TU_TDEC_MEMBER parentMember = findMemberParent(tdec, reducedMember->member);
    ReducedMember* reducedParent = newcolumn->membersToReducedMembers[parentMember];
    TU_TDEC_EDGE markerOfParent = tdec->members[member].markerOfParent;

    TUdbgMsg(6 + 2*depth, "Marker edge closes cycle.\n");
    TUdbgMsg(6 + 2*depth, "Parent member %d is reduced member %ld.\n", parentMember,
      (reducedParent - newcolumn->reducedMembers));

    /* Add marker edge of parent to reduced parent's reduced edges. */

    assert(newcolumn->numPathEdges < newcolumn->memPathEdges);
    PathEdge* reducedEdge = &newcolumn->pathEdgeStorage[newcolumn->numPathEdges];
    ++newcolumn->numPathEdges;
    reducedEdge->edge = markerOfParent;
    reducedEdge->next = reducedParent->firstPathEdge;
    reducedParent->firstPathEdge = reducedEdge;

    /* Indicate that marker edge of parent belongs to path. */
    newcolumn->edgesInPath[markerOfParent] = true;

    /* Increase node degrees of nodes in a prime parent. */
    if (tdec->members[reducedParent->member].type == TDEC_MEMBER_TYPE_PRIME)
    {
      newcolumn->nodesDegree[findEdgeHead(tdec, markerOfParent)]++;
      newcolumn->nodesDegree[findEdgeTail(tdec, markerOfParent)]++;
    }

    TUdbgMsg(6 + 2*depth, "Added marker edge of parent to list of reduced edges.\n");
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

  TUdbgMsg(0, "\n  Checking whether we can add a column with %d 1's.\n", numEntries);

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  TU_CALL( initializeNewColumn(tu, tdec, newcolumn) );
  TU_CALL( computeReducedDecomposition(tu, tdec, newcolumn, entryRows, numEntries) );
  TU_CALL( initializeReducedMemberEdgeLists(tu, tdec, newcolumn, entryRows, numEntries) );

  for (int i = 0; i < newcolumn->numReducedComponents; ++i)
  {
    TU_CALL( determineTypes(tu, tdec, newcolumn, &newcolumn->reducedComponents[i],
      newcolumn->reducedComponents[i].root, 0) );
  }

  if (newcolumn->remainsGraphic)
    TUdbgMsg(4, "Adding the column would maintain graphicness.\n");

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  return TU_OKAY;
}

static
TU_ERROR setEdgeNodes(
  TU* tu,             /**< \ref TU environment. */
  TU_TDEC* tdec,      /**< t-decomposition. */
  TU_TDEC_EDGE edge,  /**< Reduced component. */
  TU_TDEC_NODE tail,  /**< New tail node. */ 
  TU_TDEC_NODE head   /**< New head node. */
)
{
  assert(tu);
  assert(tdec);

  tdec->edges[edge].tail = tail;
  tdec->edges[edge].head = head;

  return TU_OKAY;
}

static
TU_ERROR mergeMemberIntoParent(
  TU* tu,                 /**< \ref TU environment. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_MEMBER member,  /**< Reduced component. */
  bool headToHead         /**< Whether the heads of the edges shall be joined. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);

  member = findMember(tdec, member);
  TU_TDEC_MEMBER parentMember = findMemberParent(tdec, member);
  assert(parentMember >= 0);

  TU_TDEC_EDGE parentEdge = tdec->members[member].markerOfParent;
  assert(parentEdge >= 0);
  TU_TDEC_EDGE childEdge = tdec->members[member].markerToParent;
  assert(childEdge >= 0);

  TUdbgMsg(10, "Merging member %d into %d, identifying %d with %d.\n", member, parentMember, childEdge, parentEdge);

  TU_TDEC_NODE parentEdgeNodes[2] = { findEdgeTail(tdec, parentEdge), findEdgeHead(tdec, parentEdge) };
  TU_TDEC_NODE childEdgeNodes[2] = { findEdgeTail(tdec, childEdge), findEdgeHead(tdec, childEdge) };

  /* Identify nodes. */
  
  tdec->nodes[childEdgeNodes[0]].representativeNode = parentEdgeNodes[headToHead ? 0 : 1];
  tdec->nodes[childEdgeNodes[1]].representativeNode = parentEdgeNodes[headToHead ? 1 : 0];

  /* Identify members. */

  tdec->members[member].representativeMember = parentMember;

  /* We add the member's edges to the parent's edge list and thereby remove the two marker edges. */
  if (tdec->members[parentMember].firstEdge == parentEdge)
    tdec->members[parentMember].firstEdge = tdec->edges[parentEdge].next;

  tdec->edges[tdec->edges[parentEdge].next].prev = tdec->edges[childEdge].prev;
  tdec->edges[tdec->edges[parentEdge].prev].next = tdec->edges[childEdge].next;
  tdec->edges[tdec->edges[childEdge].next].prev = tdec->edges[parentEdge].prev;
  tdec->edges[tdec->edges[childEdge].prev].next = tdec->edges[parentEdge].next;
  tdec->members[parentMember].numEdges += tdec->members[member].numEdges - 2;
  tdec->numEdges -= 2;
  tdec->edges[parentEdge].next = tdec->firstFreeEdge;
  tdec->edges[childEdge].next = parentEdge;
  tdec->firstFreeEdge = childEdge;
  tdec->members[parentMember].type = TDEC_MEMBER_TYPE_PRIME;

  return TU_OKAY;
}

static
TU_ERROR createBondNodes(
  TU* tu,               /**< \ref TU environment. */
  TU_TDEC* tdec,        /**< t-decomposition. */
  TU_TDEC_MEMBER member /**< A bond member. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);
  member = findMember(tdec, member);
  assert(tdec->members[member].type == TDEC_MEMBER_TYPE_BOND);

  TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
  if (tdec->edges[edge].head >= 0)
  {
    assert(tdec->edges[edge].tail >= 0);
    return TU_OKAY;
  }

  TU_TDEC_NODE tail, head;
  TU_CALL( createNode(tu, tdec, &tail) );
  TU_CALL( createNode(tu, tdec, &head) );

  do
  {
    assert(tdec->edges[edge].tail < 0);
    assert(tdec->edges[edge].head < 0);

    tdec->edges[edge].tail = tail;
    tdec->edges[edge].head = head;
    edge = tdec->edges[edge].next;
  }
  while (edge != tdec->members[member].firstEdge);

  return TU_OKAY;
}

static
TU_ERROR addColumnProcessBond(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int depth                           /**< Depth of this member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);
  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);
  assert(tdec->members[member].type == TDEC_MEMBER_TYPE_BOND);

  int numOneEnd, numTwoEnds;
  TU_TDEC_EDGE childMarkerEdges[2];
  TU_CALL( countChildrenTypes(tu, tdec, reducedMember, &numOneEnd, &numTwoEnds, childMarkerEdges) );
  
  TUdbgMsg(6 + 2*depth, "addColumnProcessBond for%s member %d (reduced %ld), #one-ends = %d, #two-ends = %d.\n",
    depth == 0 ? " root" : "", member, (reducedMember - newcolumn->reducedMembers), numOneEnd, numTwoEnds);

  if (depth == 0)
  {
    if (numOneEnd == 0 && numTwoEnds == 0)
    {
      assert(reducedMember->firstPathEdge);
      TU_CALL( addTerminal(tu, tdec, reducedComponent, member, -1) );
      TU_CALL( addTerminal(tu, tdec, reducedComponent, member, -1) );

      return TU_OKAY;
    }
    else if (numOneEnd == 1)
    {
      assert(reducedComponent->numTerminals == 1);

      /* We don't merge since the child shall be the terminal member (rule (R3) in the paper)! */
      assert(reducedMember->firstPathEdge);
      TU_TDEC_MEMBER childMember = tdec->edges[childMarkerEdges[0]].childMember;
      TU_TDEC_EDGE childsParentMarker = tdec->members[childMember].markerToParent;
      TU_CALL( addTerminal(tu, tdec, reducedComponent, childMember, findEdgeTail(tdec, childsParentMarker)) );
      tdec->members[childMember].type = TDEC_MEMBER_TYPE_PRIME;
    }
    else if (numOneEnd == 2)
    {
      assert(reducedComponent->numTerminals == 2);

      if (tdec->members[member].numEdges > 3)
      {
        TU_TDEC_MEMBER newBond;
        TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_BOND, &newBond) );
        TU_TDEC_EDGE markerOfParentBond, markerOfChildBond;
        TU_CALL( createEdge(tu, tdec, member, &markerOfParentBond) );
        TU_CALL( addEdgeToMembersEdgeList(tu, tdec, markerOfParentBond, member) );
        TU_CALL( createEdge(tu, tdec, newBond, &markerOfChildBond) );
        TU_CALL( addEdgeToMembersEdgeList(tu, tdec, markerOfChildBond, newBond) );
        tdec->members[newBond].markerOfParent = markerOfParentBond;
        tdec->members[newBond].markerToParent = markerOfChildBond;
        tdec->members[newBond].parentMember = member;
        tdec->edges[markerOfParentBond].childMember = newBond;
        tdec->edges[markerOfChildBond].childMember = -1;

        for (int c = 0; c < 2; ++c)
        {
          TU_CALL( removeEdgeFromMembersEdgeList(tu, tdec, childMarkerEdges[c], member) );
          tdec->edges[childMarkerEdges[c]].member = newBond;
          TU_CALL( addEdgeToMembersEdgeList(tu, tdec, childMarkerEdges[c], newBond) );
          tdec->members[findMember(tdec, tdec->edges[childMarkerEdges[c]].childMember)].parentMember = newBond;
        }

        member = newBond;
        reducedMember->member = member;
      }

#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

      assert(tdec->members[member].numEdges == 3);
      TU_CALL( createBondNodes(tu, tdec, member) );
      TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[0]].childMember, true) );
      TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[1]].childMember,
        reducedMember->firstPathEdge == NULL) );

#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

      return TU_OKAY;
    }
    else
    {
      assert(0 == "addColumnProcessBond for root not fully implemented.");
    }
  }
  else
  {
    /* Non-root bond.*/

    if (numOneEnd == 1)
    {
      assert(reducedComponent->numTerminals >= 1);

      if (tdec->members[member].numEdges >= 4)
      {
        assert(0 == "addColumnProcessBond for root with one 1-end and >= 4 edges is not implemented: we need to split off the two child marker edges into another bond.");
      }

      TU_TDEC_NODE tail, head;
      TU_CALL( createNode(tu, tdec, &tail) );
      TU_CALL( createNode(tu, tdec, &head) );
      TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
      do 
      {
        TU_CALL( setEdgeNodes(tu, tdec, edge, tail, head) );
        edge = tdec->edges[edge].next;
      }
      while (edge != tdec->members[member].firstEdge);

      TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[0]].childMember,
        !reducedMember->firstPathEdge) );

#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

      return TU_OKAY;
    }
    else
    {
      assert(0 == "addColumnProcessBond is not fully implemented.");
    }
  }

  return TU_OKAY;
}

// TODO: inline
static void flipEdge(
  TU_TDEC* tdec,    /**< t-decomposition. */
  TU_TDEC_EDGE edge /**< edge. */
)
{
  assert(tdec);
  assert(edge >= 0);
  assert(edge < tdec->memEdges);

  TU_TDEC_NODE tmp = tdec->edges[edge].head;
  tdec->edges[edge].head = tdec->edges[edge].tail;
  tdec->edges[edge].tail = tmp;
}

static
TU_ERROR addColumnProcessPrime(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int depth                           /**< Depth of this member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);
  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);
  assert(tdec->members[member].type == TDEC_MEMBER_TYPE_PRIME);

  int numOneEnd, numTwoEnds;
  TU_TDEC_EDGE childMarkerEdges[2];
  TU_CALL( countChildrenTypes(tu, tdec, reducedMember, &numOneEnd, &numTwoEnds, childMarkerEdges) );

  TUdbgMsg(6 + 2*depth, "addColumnProcessPrime for%s member %d (reduced %ld), #one-ends = %d, #two-ends = %d.\n",
    depth == 0 ? " root" : "", reducedMember->member, (reducedMember - newcolumn->reducedMembers), numOneEnd,
    numTwoEnds);

  TU_TDEC_NODE parentMarkerNodes[2] = {
    depth == 0 ? -1 : findEdgeTail(tdec, tdec->members[member].markerToParent),
    depth == 0 ? -1 : findEdgeHead(tdec, tdec->members[member].markerToParent)
  };
  TU_TDEC_NODE childMarkerNodes[4] = {
    childMarkerEdges[0] < 0 ? -1 : findEdgeTail(tdec, childMarkerEdges[0]),
    childMarkerEdges[0] < 0 ? -1 : findEdgeHead(tdec, childMarkerEdges[0]),
    childMarkerEdges[1] < 0 ? -1 : findEdgeTail(tdec, childMarkerEdges[1]),
    childMarkerEdges[1] < 0 ? -1 : findEdgeHead(tdec, childMarkerEdges[1])
  };

  if (depth == 0)
  {
    /* Root prime. */
    if (numOneEnd == 0 && numTwoEnds == 0)
    {
      assert(reducedComponent->numTerminals == 2);
    }
    else
    {
      assert(0 == "addColumnProcessPrime is not fully implemented.");
    }
  }
  else
  {
    /* Non-root prime. */

    if (numOneEnd == 0 && numTwoEnds == 0)
    {
      assert(reducedComponent->numTerminals >= 1);
      assert(reducedMember->firstPathEdge);
      assert(reducedMember->primeEndNodes[0] >= 0);

      if (parentMarkerNodes[0] == reducedMember->primeEndNodes[0])
        flipEdge(tdec, tdec->members[member].markerToParent);
    }
    else if (numOneEnd == 1)
    {
      if (reducedMember->primeEndNodes[0] >= 0)
      {
        /* primeEndNodes[0] is the connecting node of the parent marker. */
        /* primeEndNodes[1] is the connecting node of the child marker. */
        assert(reducedMember->primeEndNodes[1] >= 0);
        
        assert(0 == "addColumnProcessPrime is not fully implemented.");
      }
      else
      {
        /* Parent marker and child marker must be next to each other. */
        if (parentMarkerNodes[0] == childMarkerNodes[0] || parentMarkerNodes[0] == childMarkerNodes[1])
          flipEdge(tdec, tdec->members[member].markerToParent);

        TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[0]].childMember,
          parentMarkerNodes[0] == childMarkerNodes[1] || parentMarkerNodes[1] == childMarkerNodes[1]) );

#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
    }
    else
    {
      assert(0 == "addColumnProcessPrime is not fully implemented.");
    }
  }

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
  fflush(stdout);
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

  tdec->edges[edge].member = bond;
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
  TU_TDEC_MEMBER member,        /**< Polygon member to be squeezed. */
  bool* edgesPredicate,         /**< Map from edges to predicate. */
  bool predicateValue,          /**< Value of predicate. */
  TU_TDEC_MEMBER* pNewBond,     /**< Pointer for storing the new bond. */
  TU_TDEC_MEMBER* pNewPolygon   /**< Pointer for storing the new polygon. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);
  assert(member < tdec->memMembers);
  assert(edgesPredicate);
  assert(tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON);

#if defined(TU_DEBUG_TDEC)
  TUdbgMsg(8, "Squeezing polygon %d.\n", member);
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

#if !defined(NDEBUG)

  /* We are only supposed to squeeze paths of length at least 2. */

  TU_TDEC_EDGE e = tdec->members[member].firstEdge;
  int numSatisfying = 0;
  do
  {
    bool value = edgesPredicate[e];
    if ((value && predicateValue) || (!value && !predicateValue))
      ++numSatisfying;
    e = tdec->edges[e].next;
  } while (e != tdec->members[member].firstEdge);
  assert(numSatisfying > 1);
#endif /* !NDEBUG */
  
  /* Go through old polygon. */

  TU_TDEC_EDGE firstEdge = tdec->members[member].firstEdge;
  TU_TDEC_EDGE edge = firstEdge;
  bool encounteredStayingEdge = false;
  do
  {
#if defined(TU_DEBUG_TDEC_SQUEEZE)
    TUdbgMsg(8, "Edge %d <%d>", edge, tdec->edges[edge].name);
    if (tdec->edges[edge].childMember >= 0)
      TUdbgMsg(0, " (with child %d)", tdec->edges[edge].childMember);
    if (edge == tdec->members[member].markerToParent)
      TUdbgMsg(0, " (with parent %d)", tdec->members[member].parentMember);
    TUdbgMsg(0, " (prev = %d, next = %d)", tdec->edges[edge].prev, tdec->edges[edge].next);
#endif /* TU_DEBUG_TDEC */
    /* Evaluate predicate. */
    bool value = edgesPredicate[edge];
    if ((value && !predicateValue) || (!value && predicateValue))
    {
#if defined(TU_DEBUG_TDEC_SQUEEZE)
      TUdbgMsg(" does not satisfy the predicate.\n");
#endif /* TU_DEBUG_TDEC */
      edge = tdec->edges[edge].next;
      encounteredStayingEdge = true;
      continue;
    }

#if defined(TU_DEBUG_TDEC_SQUEEZE)
    TUdbgMsg(0, " satisfies the predicate.\n");
#endif /* TU_DEBUG_TDEC */

    assert(edge != tdec->members[member].markerToParent);

    /* Remove edge from old edge list. */
    TU_TDEC_EDGE oldPrev = tdec->edges[edge].prev;
    TU_TDEC_EDGE oldNext = tdec->edges[edge].next;
    tdec->edges[oldPrev].next = oldNext;
    tdec->edges[oldNext].prev = oldPrev;
    tdec->members[member].numEdges--;

    /* Add edge to new edge list. */
    TU_TDEC_EDGE newPrev = tdec->edges[polygonParentMarker].prev;
    tdec->edges[newPrev].next = edge;
    tdec->edges[polygonParentMarker].prev = edge;
    tdec->edges[edge].prev = newPrev;
    tdec->edges[edge].next = polygonParentMarker;
    tdec->edges[edge].member = polygon;
    if (tdec->edges[edge].childMember >= 0)
    {
      assert( tdec->members[tdec->edges[edge].childMember].parentMember == member);
      tdec->members[tdec->edges[edge].childMember].parentMember = polygon;
    }
    tdec->members[polygon].numEdges++;

    /* Did we move the first edge of this member? */
    if (edge == firstEdge)
    {
      tdec->members[member].firstEdge = oldNext;
      firstEdge = oldNext;
      edge = oldNext;
      continue;
    }

    edge = oldNext;
  }
  while (edge != firstEdge || !encounteredStayingEdge);

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
  tdec->members[member].numEdges++;

  /* Link all. */
  tdec->members[polygon].parentMember = bond;
  tdec->edges[bondChildMarker].childMember = polygon;
  tdec->members[bond].parentMember = member;
  tdec->edges[memberChildMarker].childMember = bond;

#if defined(TU_DEBUG_TDEC_SQUEEZE)
  TUdbgMsg(8, "Updated old polygon with these edges:");
  edge = firstEdge;
  do
  {
    TUdbgMsg(0, " %d <%d>", edge, tdec->edges[edge].name);
    edge = tdec->edges[edge].next;
  }
  while (edge != firstEdge);
  TUdbgMsg(0, ".\n");
  TUdbgMsg(8, "New polygon has these edges:");
  edge = polygonParentMarker;
  do
  {
    TUdbgMsg(0, " %d <%d>", edge, tdec->edges[edge].name);
    edge = tdec->edges[edge].next;
  }
  while (edge != polygonParentMarker);
  TUdbgMsg(0, ".\n");
#endif /* TU_DEBUG_TDEC */

  TUdbgMsg(8, "Connecting bond is member %d and squeezed polygon is member %d.\n", bond, polygon);

  if (pNewBond)
    *pNewBond = bond;
  if (pNewPolygon)
    *pNewPolygon = polygon;

  return TU_OKAY;
}

static
TU_ERROR addColumnProcessPolygon(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int depth                           /**< Depth of this member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedMember);

  TU_TDEC_MEMBER member = findMember(tdec, reducedMember->member);

  int numOneEnd, numTwoEnds;
  TU_TDEC_EDGE childMarkerEdges[2];
  TU_CALL( countChildrenTypes(tu, tdec, reducedMember, &numOneEnd, &numTwoEnds, childMarkerEdges) );

  TUdbgMsg(6 + 2*depth, "addColumnProcessPolygon for%s member %d (reduced %ld), #one-ends = %d, #two-ends = %d.\n",
    depth == 0 ? " root" : "", member, (reducedMember - newcolumn->reducedMembers), numOneEnd, numTwoEnds);

  if (depth == 0)
  {
    if (numOneEnd == 0 && numTwoEnds == 0)
    {
      /* Root polygon containing both ends. */

      assert(reducedMember->firstPathEdge);
      TU_TDEC_MEMBER newBond;
      if (reducedMember->firstPathEdge->next == NULL)
      {
        /* There is only one path edge, so we create a bond for that edge. */
        TU_TDEC_EDGE bondChildMarker;
        TU_CALL( createEdgeBond(tu, tdec, newcolumn, member, reducedMember->firstPathEdge->edge, &bondChildMarker) );
        newBond = tdec->edges[bondChildMarker].childMember;
        // TODO: maybe change createEdgeBond to return pointer to bond.
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
      else
      {
        /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
         * it to the remaining polygon. */
 
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, true, &newBond, NULL) );
      }
      TU_CALL( addTerminal(tu, tdec, reducedComponent, newBond, -1) );
      TU_CALL( addTerminal(tu, tdec, reducedComponent, newBond, -1) );
      
#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
    }
    else if (numOneEnd == 1)
    {
      /* Root polygon containing one end. */

      assert(reducedMember->firstPathEdge);

      /* If there is more than 1 path edge, we squeeze off by moving them to a new polygon and creating a bond to
       * connect it to the remaining polygon. */
      TU_TDEC_EDGE pathEdge;
      if (reducedMember->firstPathEdge->next == NULL)
        pathEdge = reducedMember->firstPathEdge->edge;
      else
      {
        TU_TDEC_MEMBER newBond = -1;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, true, &newBond, NULL) );

        pathEdge = tdec->members[newBond].markerOfParent;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }

      /* Unless the polygon consists of only the parent marker, the child marker (containing a path end) and a
       * representative edge, we squeeze off the representative edge and the child marker. */
      if (tdec->members[member].numEdges > 3)
      {
        newcolumn->edgesInPath[childMarkerEdges[0]] = true;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, true, NULL, &member) );
        newcolumn->edgesInPath[childMarkerEdges[0]] = false;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }

      assert(tdec->members[member].numEdges == 3);

      TU_TDEC_NODE a, b, c;
      TU_CALL( createNode(tu, tdec, &a) );
      TU_CALL( createNode(tu, tdec, &b) );
      TU_CALL( createNode(tu, tdec, &c) );
      TU_CALL( setEdgeNodes(tu, tdec, tdec->members[member].markerToParent, b, c) );
      TU_CALL( setEdgeNodes(tu, tdec, pathEdge, a, b) );
      TU_CALL( setEdgeNodes(tu, tdec, childMarkerEdges[0], c, a) );
      TU_CALL( addTerminal(tu, tdec, reducedComponent, member, b) );
      TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[0]].childMember, true) );

#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
    }
    else
    {
      assert(0 == "addColumnProcessPolygon for root with interesting children not implemented.");
    }
  }
  else
  {
    if (reducedMember->type == TYPE_3_EXTENSION && numOneEnd + numTwoEnds == 0)
    {
      /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
      * it to the remaining polygon. */

      assert(reducedComponent->numTerminals < 2);
      assert(reducedMember->firstPathEdge);

      /* Squeeze off path edges (if more than one). */
      if (reducedMember->firstPathEdge->next)
      {
        TU_TDEC_MEMBER newBond;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, true, &newBond, NULL) );
        reducedMember->polygonPathEdge = tdec->members[newBond].markerOfParent;
        newcolumn->edgesInPath[reducedMember->polygonPathEdge] = true;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
      else
      {
        reducedMember->polygonPathEdge = reducedMember->firstPathEdge->edge;
      }
      
      /* If necessary, we squeeze off the non-path edges as well. */

      assert(tdec->members[member].numEdges >= 3);
      if (tdec->members[member].numEdges == 3)
      {
        reducedMember->polygonNonpathEdge = tdec->edges[tdec->members[member].markerToParent].next;
        if (reducedMember->polygonNonpathEdge == reducedMember->polygonPathEdge)
          reducedMember->polygonNonpathEdge = tdec->edges[reducedMember->polygonNonpathEdge].next;
      }
      else
      {
        /* We temporarily mark the parent edge to belong to the path. */
        TU_TDEC_EDGE markerToParent = tdec->members[member].markerToParent;
        newcolumn->edgesInPath[markerToParent] = true;
        TU_TDEC_MEMBER newBond;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, false, &newBond, NULL) );
        reducedMember->polygonNonpathEdge = tdec->members[newBond].markerOfParent;
        newcolumn->edgesInPath[markerToParent] = false;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
      assert(tdec->members[member].numEdges == 3);

      /* We now create the nodes of the triangle so that the path leaves it via the parent marker edge's head node. */

      TU_TDEC_NODE a, b, c;
      TU_CALL( createNode(tu, tdec, &a) );
      TU_CALL( createNode(tu, tdec, &b) );
      TU_CALL( createNode(tu, tdec, &c) );
      TU_CALL( setEdgeNodes(tu, tdec, tdec->members[reducedMember->member].markerToParent, a, b) );
      TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonPathEdge, b, c) );
      TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonNonpathEdge, c, a) );
      TU_CALL( addTerminal(tu, tdec, reducedComponent, reducedMember->member, c) );

      return TU_OKAY;
    }
    else if (reducedMember->type == TYPE_3_EXTENSION && numOneEnd == 1)
    {
      /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
      * it to the remaining polygon. */
      if (!reducedMember->firstPathEdge)
        reducedMember->polygonPathEdge = -1;
      else if (!reducedMember->firstPathEdge->next)
        reducedMember->polygonPathEdge = reducedMember->firstPathEdge->edge;
      else
      {
        TU_TDEC_MEMBER newBond;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, true, &newBond, NULL) );
        reducedMember->polygonPathEdge = tdec->members[newBond].markerOfParent;
        newcolumn->edgesInPath[reducedMember->polygonPathEdge] = true;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
    
      /* If necessary, we squeeze off the non-path edges as well. */
      assert(tdec->members[member].numEdges >= 3);      
      if (tdec->members[member].numEdges == 3)
        reducedMember->polygonNonpathEdge = -1;
      else if (tdec->members[member].numEdges == (reducedMember->polygonPathEdge >= 0 ? 4 : 3))
      {
        reducedMember->polygonNonpathEdge = tdec->members[member].firstEdge;
        while (reducedMember->polygonNonpathEdge == tdec->members[member].markerToParent
          || reducedMember->polygonNonpathEdge == childMarkerEdges[0])
        {
          reducedMember->polygonNonpathEdge = tdec->edges[reducedMember->polygonNonpathEdge].next;
        }
      }
      else
      {
        newcolumn->edgesInPath[tdec->members[member].markerToParent] = true;
        newcolumn->edgesInPath[childMarkerEdges[0]] = true;
        TU_TDEC_MEMBER newBond;
        TU_CALL( squeezePolygonEdges(tu, tdec, member, newcolumn->edgesInPath, false, &newBond, NULL) );
        reducedMember->polygonNonpathEdge = tdec->members[newBond].markerOfParent;
        newcolumn->edgesInPath[tdec->members[member].markerToParent] = false;
        newcolumn->edgesInPath[childMarkerEdges[0]] = false;
        
#if defined(TU_DEBUG_DOT)
        TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
      }
      
      assert(tdec->members[member].numEdges <= 4);

      /* We now create the nodes of the triangle so that the path leaves it via the parent marker edge's head node. */

      TU_TDEC_NODE a, b, c, d;
      TU_CALL( createNode(tu, tdec, &a) );
      TU_CALL( createNode(tu, tdec, &b) );
      TU_CALL( createNode(tu, tdec, &c) );
      TU_CALL( setEdgeNodes(tu, tdec, tdec->members[reducedMember->member].markerToParent, a, b) );
      if (tdec->members[member].numEdges == 4)
      {
        TU_CALL( createNode(tu, tdec, &d) );
        TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonPathEdge, b, c) );
        TU_CALL( setEdgeNodes(tu, tdec, childMarkerEdges[0], d, c) );
        TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonNonpathEdge, d, a) );
      }
      else if (reducedMember->polygonNonpathEdge == -1)
      {
        TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonPathEdge, b, c) );
        TU_CALL( setEdgeNodes(tu, tdec, childMarkerEdges[0], a, c) );
      }
      else
      {
        assert(reducedMember->polygonPathEdge == -1);
        TU_CALL( setEdgeNodes(tu, tdec, childMarkerEdges[0], c, b) );
        TU_CALL( setEdgeNodes(tu, tdec, reducedMember->polygonNonpathEdge, a, c) );
      }
      TU_CALL( mergeMemberIntoParent(tu, tdec, tdec->edges[childMarkerEdges[0]].childMember, true) );

#if defined(TU_DEBUG_DOT)
      TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

      return TU_OKAY;
    }
    
    assert(0 == "addColumnProcessPolygon for non-root not implemented.");
  }
  
  
//   if (depth == 0 && numOneEnd == 0 && numTwoEnds == 0)
//   {
//     /* Root polygon containing both ends. */
// 
//     assert(reducedMember->firstPathEdge);
//     if (reducedMember->firstPathEdge->next == NULL)
//     {
//       /* There is only one path edge, so we create a bond for that edge. */
//       TU_TDEC_EDGE bondChildMarker;
//       TU_CALL( createEdgeBond(tu, tdec, newcolumn, reducedMember->member,
//         reducedMember->firstPathEdge->edge, &bondChildMarker) );
// 
//       TU_TDEC_MEMBER bond = tdec->edges[bondChildMarker].childMember;
//       reducedComponent->terminalMember[0] = bond;
//       reducedComponent->terminalMember[1] = bond;
//       reducedComponent->numTerminals = 2;
//       return TU_OKAY;
//     }
//     else
//     {
//       /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
//        * it to the remaining polygon. */
// 
//       TU_TDEC_MEMBER newBond = -1;
//       TU_CALL( squeezePolygonEdges(tu, tdec, reducedMember->member, newcolumn->edgesInPath, true, &newBond, NULL) );
// 
//       TU_TDEC_MEMBER bond = tdec->edges[tdec->members[newBond].markerOfParent].childMember;
//       reducedComponent->terminalMember[0] = bond;
//       reducedComponent->terminalMember[1] = bond;
//       reducedComponent->numTerminals = 2;
//       return TU_OKAY;
//     }
//   }
//   else if (depth > 0 && reducedMember->type == TYPE_3_EXTENSION && numOneEnd + numTwoEnds == 0)
//   {
//     /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
//      * it to the remaining polygon. */
// 
//     assert(reducedComponent->numTerminals < 2);
//     assert(reducedMember->firstPathEdge);
// 
//     /* Squeeze off path edges (if more than one). */
//     if (reducedMember->firstPathEdge->next)
//     {
//       TU_TDEC_MEMBER newBond;
//       TU_CALL( squeezePolygonEdges(tu, tdec, reducedMember->member, newcolumn->edgesInPath, true, &newBond, NULL) );
//       reducedMember->polygonPathEdge = tdec->members[newBond].markerOfParent;
//       newcolumn->edgesInPath[reducedMember->polygonPathEdge] = true;
//     }
//     else
//     {
//       reducedMember->polygonPathEdge = reducedMember->firstPathEdge->edge;
//     }
// 
// //     printf("squeezedPathEdge = %d. Old polygon has length %d\n", reducedMember->representativePathEdge, tdec->members[reducedMember->member].numEdges);
// 
//     assert(tdec->members[reducedMember->member].numEdges >= 3);
//     if (tdec->members[reducedMember->member].numEdges == 3)
//     {
//       reducedMember->polygonNonpathEdge = tdec->edges[tdec->members[reducedMember->member].markerToParent].next;
//       if (reducedMember->polygonNonpathEdge == reducedMember->polygonPathEdge)
//         reducedMember->polygonNonpathEdge = tdec->edges[reducedMember->polygonNonpathEdge].next;
//     }
//     else
//     {
//       /* We temporarily mark the parent edge to belong to the path. */
//       TU_TDEC_EDGE markerToParent = tdec->members[reducedMember->member].markerToParent;
//       newcolumn->edgesInPath[markerToParent] = true;
//       TU_TDEC_MEMBER newBond;
//       TU_CALL( squeezePolygonEdges(tu, tdec, reducedMember->member, newcolumn->edgesInPath, false, &newBond, NULL) );
//       reducedMember->polygonNonpathEdge = tdec->members[newBond].markerOfParent;
//       newcolumn->edgesInPath[markerToParent] = false;
//     }
//     assert(tdec->members[reducedMember->member].numEdges == 3);
// 
//     TU_CALL( addTerminal(tu, tdec, reducedComponent, reducedMember->member, -1) );
// 
//     return TU_OKAY;
//   }
//   else if (depth == 0 && numOneEnd == 1)
//   {
//     assert(numTwoEnds == 0);
// 
//     /* Squeeze off all path edges by moving them to a new polygon and creating a bond to connect
//      * it to the remaining polygon. */
// 
//     assert(reducedComponent->numTerminals < 2);
//     assert(reducedMember->firstPathEdge);
// 
//     /* Squeeze off path edges (if more than one). */
//     if (reducedMember->firstPathEdge->next)
//     {
//       TU_TDEC_MEMBER newBond;
//       TU_CALL( squeezePolygonEdges(tu, tdec, reducedMember->member, newcolumn->edgesInPath, true, &newBond, NULL) );
//       reducedMember->polygonPathEdge = tdec->members[newBond].markerOfParent;
//       newcolumn->edgesInPath[reducedMember->polygonPathEdge] = true;
//     }
//     else
//     {
//       reducedMember->polygonPathEdge = reducedMember->firstPathEdge->edge;
//     }
// 
//     /* We find the child marker to the child containing the other path end. */
//     TU_TDEC_EDGE childMarkerEdge = -1;
//     for (int c = 0; c < reducedMember->numChildren; ++c)
//     {
//       Type childType = reducedMember->children[c]->type;
//       if (childType == TYPE_2_SHORTCUT || childType == TYPE_3_EXTENSION)
//       {
//         TU_TDEC_MEMBER childMember = findMember(tdec,  reducedMember->children[c]->member);
//         childMarkerEdge = tdec->members[childMember].markerOfParent;
//       }
//     }
//     assert(childMarkerEdge >= 0);
// 
//     /* 
//      * Unless the polygon consists of only the parent marker, the child marker (containing a path end) and a
//      * representative edge, we squeeze off the representative edge and the child marker.
//      */
//     if (tdec->members[findMember(tdec, reducedMember->member)].numEdges > 3)
//     {
//       newcolumn->edgesInPath[childMarkerEdge] = true;
//       TU_TDEC_MEMBER newBond, newPolygon;
//       TU_CALL( squeezePolygonEdges(tu, tdec, reducedMember->member, newcolumn->edgesInPath, true, &newBond,
//         &newPolygon) );
//       newcolumn->edgesInPath[childMarkerEdge] = false;
// 
// #if defined(TU_DEBUG_TDEC)
//       printf("        Reduced member %d is replaced by new polygon %d.\n", reducedMember->member, newPolygon);
//       fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
//       reducedMember->member = newPolygon;
//       newcolumn->membersToReducedMembers[newPolygon] = reducedMember;
//       
//       /* Remove those children from reduced member that are not children anymore. */
//       for (int c = 0; c < reducedMember->numChildren; )
//       {
//         if (findMemberParent(tdec, reducedMember->children[c]->member) == newPolygon)
//           ++c;
//         else
//         {
// #if defined(TU_DEBUG_TDEC)
//           printf("          Removing child member %d.\n", reducedMember->children[c]->member);
//           fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//           reducedMember->children[c] = reducedMember->children[reducedMember->numChildren-1];
//           reducedMember->numChildren--;
//         }
//       }
//     }
// 
//     reducedComponent->terminalMember[reducedComponent->numTerminals] = reducedMember->member;
//     reducedComponent->numTerminals++;
// 
//     return TU_OKAY;
//   }
// 
// #if defined(TU_DEBUG_TDEC)
//   printf("        End of addColumnPreprocessPolygon for reduced%s member %ld (member %d), #one-ends = %d, #two-ends = %d.\n",
//     depth == 0 ? " root" : "", (reducedMember - newcolumn->reducedMembers), reducedMember->member,
//     numOneEnd, numTwoEnds);
//   fflush(stdout);
// #endif /* TU_DEBUG_TDEC */


//   assert(0 == "addColumnPreprocessPolygon is not fully implemented.");

  return TU_OKAY;
}

/**
 * \brief Process reduced t-decomposition before the actual modification.
 * 
 * Processes the reduced members in depth-first search manner and does the following:
 * - Polygons are squeezed.
 * - Terminal nodes and (reduced) members are detected.
 * - Marker edges along unique path between terminal nodes are merged.
 */

static
TU_ERROR addColumnProcessComponent(
  TU* tu,                             /**< \ref TU environment. */
  TU_TDEC* tdec,                      /**< t-decomposition. */
  TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
  ReducedComponent* reducedComponent, /**< Reduced component. */
  ReducedMember* reducedMember,       /**< Reduced member. */
  int depth                           /**< Depth of member in reduced t-decomposition. */
)
{
  assert(tu);
  assert(tdec);
  assert(newcolumn);
  assert(reducedComponent);

  TUdbgMsg(6 + 2*depth, "addColumnProcess(member %d = reduced member %ld)\n", reducedMember->member,
    (reducedMember - &newcolumn->reducedMembers[0]));

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );
  
  /* If we are type 1, then we don't need to do anything. */
  if (reducedMember->type == TYPE_1_CLOSES_CYCLE)
  {
    return TU_OKAY;
  }

  /* Handle children recursively. */
  for (int c = 0; c < reducedMember->numChildren; ++c)
  {
    ReducedMember* child = reducedMember->children[c];
    if (child->type != TYPE_1_CLOSES_CYCLE)
    {
      TU_CALL( addColumnProcessComponent(tu, tdec, newcolumn, reducedComponent, reducedMember->children[c], depth+1) );
    }
    else
    {
      TUdbgMsg(8 + 2*depth, "Member %d is implicitly replaced by a path edge.\n", findMember(tdec, child->member));
    }
  }

  /* Different behavior for bonds, polygons and prime components. */
  if (tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_BOND)
    TU_CALL( addColumnProcessBond(tu, tdec, newcolumn, reducedComponent, reducedMember, depth) );
  else if (tdec->members[reducedMember->member].type == TDEC_MEMBER_TYPE_POLYGON)
    TU_CALL( addColumnProcessPolygon(tu, tdec, newcolumn, reducedComponent, reducedMember, depth) );
  else
    TU_CALL( addColumnProcessPrime(tu, tdec, newcolumn, reducedComponent, reducedMember, depth) );

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  return TU_OKAY;
}

//TODO: inline
// static TU_TDEC_NODE findCommonNode(TU_TDEC* tdec, TU_TDEC_EDGE e, TU_TDEC_EDGE f)
// {
//   TU_TDEC_NODE eHead = findEdgeHead(tdec, e);
//   TU_TDEC_NODE eTail = findEdgeTail(tdec, e);
//   TU_TDEC_NODE fHead = findEdgeHead(tdec, f);
//   TU_TDEC_NODE fTail = findEdgeTail(tdec, f);
//   if (eHead == fHead || eHead == fTail)
//     return eHead;
//   else if (eTail == fHead || eTail == fTail)
//     return eTail;
//   else
//     return -1;
// }

// static
// TU_ERROR createMissingNodes(
//   TU* tu,                             /**< \ref TU environment. */
//   TU_TDEC* tdec,                      /**< t-decomposition. */
//   TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
//   ReducedComponent* reducedComponent, /**< Reduced component. */
//   TU_TDEC_MEMBER member               /**< Member to process. */
// )
// {
//   assert(tu);
//   assert(tdec);
//   assert(newcolumn);
//   assert(reducedComponent);
// 
//   assert(isRepresentativeMember(tdec, member));
// 
//   /* We iterate to next parent until we have processed the root. */
//   while (true)
//   {
// #if defined(TU_DEBUG_TDEC)
//     printf("        Creating missing nodes of member %d.\n", member);
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     if (tdec->members[member].type == TDEC_MEMBER_TYPE_BOND)
//     {
//       TU_TDEC_EDGE edge = tdec->members[member].firstEdge;
//       if (tdec->edges[edge].head < 0)
//       {
//         assert(tdec->edges[edge].tail < 0);
//         TU_TDEC_NODE head, tail;
//         TU_CALL( createNode(tu, tdec, &head) );
//         TU_CALL( createNode(tu, tdec, &tail) );
// #if defined(TU_DEBUG_TDEC)
//         printf("          Nodes of this bond are %d and %d.\n", head, tail);
//         fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//         do
//         {
//           tdec->edges[edge].head = head;
//           tdec->edges[edge].tail = tail;
//           edge = tdec->edges[edge].next;
//         }
//         while (edge != tdec->members[member].firstEdge);
// 
//         /* When creating a new prime component, terminal nodes of bonds are always heads. */
//         if (reducedComponent->terminalMember[0] == member)
//         {
// #if defined(TU_DEBUG_TDEC)
//           printf("          Setting node %d as a terminal node.\n", head);
// #endif /* TU_DEBUG_TDEC */
//           reducedComponent->terminalNode[0] = head;
//         }
//         else if (reducedComponent->terminalMember[1] == member)
//         {
// #if defined(TU_DEBUG_TDEC)
//           printf("          Setting node %d as a terminal node.\n", head);
// #endif /* TU_DEBUG_TDEC */
//           reducedComponent->terminalNode[1] = head;
//         }
//       }
//       else
//       {
//         assert(tdec->edges[edge].tail >= 0);
//       }
//     }
//     else if (tdec->members[member].type == TDEC_MEMBER_TYPE_POLYGON)
//     {
//       TU_TDEC_EDGE firstEdge = tdec->members[member].firstEdge;
//       if (tdec->edges[firstEdge].head < 0)
//       {
//         assert(tdec->edges[firstEdge].tail < 0);
//         TU_TDEC_NODE lastHead = -1;
//         TU_TDEC_EDGE edge = firstEdge;
//         do
//         {
//           tdec->edges[edge].tail = lastHead;
//           TU_CALL( createNode(tu, tdec, &lastHead) );
//           tdec->edges[edge].head = lastHead;
//           edge = tdec->edges[edge].next;
//         }
//         while (edge != firstEdge);
//         tdec->edges[firstEdge].tail = lastHead;
// 
//         /* Determine the terminal node if this is a terminal member. */
//         TU_TDEC_NODE* pTerminalNode = NULL;
//         if (reducedComponent->terminalMember[0] == member)
//           pTerminalNode = &reducedComponent->terminalNode[0];
//         else if (reducedComponent->terminalMember[1] == member)
//           pTerminalNode = &reducedComponent->terminalNode[1];
//         if (pTerminalNode)
//         {
//           assert(tdec->members[member].numEdges == 3);
// 
//           TU_TDEC_EDGE markerToParent = tdec->members[member].markerToParent;
//           TU_TDEC_EDGE edge = tdec->edges[markerToParent].next;
//           TU_TDEC_EDGE pathEdge = newcolumn->edgesInPath[edge] ? edge : tdec->edges[edge].next;
//           TU_TDEC_EDGE nonPathEdge = newcolumn->edgesInPath[edge] ? tdec->edges[edge].next : edge;
//             
//           if (member == reducedComponent->root->member)
//             *pTerminalNode = findCommonNode(tdec, pathEdge, markerToParent);
//           else
//             *pTerminalNode = findCommonNode(tdec, pathEdge, nonPathEdge);
// #if defined(TU_DEBUG_TDEC)
//           printf("          Setting node %d as a terminal node.\n", *pTerminalNode);
// #endif /* TU_DEBUG_TDEC */
//         }
// 
// #if defined(TU_DEBUG_TDEC)
//         printf("          Edges of this polygon have these nodes:\n");
//         edge = firstEdge;
//         do 
//         {
//           printf("            %d <%d> is {%d,%d}\n", edge, tdec->edges[edge].name,
//             tdec->edges[edge].tail, tdec->edges[edge].head);
//           edge = tdec->edges[edge].next;
//         }
//         while (edge != firstEdge);
//         fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
//       }
//     }
// 
//     /* This was the reduced root, so we stop. */
//     if (member == reducedComponent->root->member)
//       break;
// 
//     member = findMemberParent(tdec, member);
//   }
// 
//   return TU_OKAY;
// }

/**
 * \brief Merges two members on a path, which finally yields a prime member.
 * 
 * If the parent of the merge is a bond, we always merge so that the path goes through the head node of the bond.
 * Hence, if the child of the merge is a bond, the path-containing node of the parent with the tail (resp. head) node
 * of the bond if the bond contains (resp. does not contain) a path edge.
 */

// static
// TU_ERROR mergeMembers(
//   TU* tu,                             /**< \ref TU environment. */
//   TU_TDEC* tdec,                      /**< t-decomposition. */
//   TU_TDEC_NEWCOLUMN* newcolumn,       /**< new column. */
//   ReducedComponent* reducedComponent, /**< Reduced component. */
//   ReducedMember* reducedMember2   /**< Reduced member. */
// )
// {
//   assert(tu);
//   assert(tdec);
//   assert(newcolumn);
//   assert(reducedComponent);
//   ReducedMember* childReducedMember = reducedMember2;
// 
//   /* We iterate to next parent until we have processed the root. */
//   while (childReducedMember != reducedComponent->root)
//   {
//     TU_TDEC_MEMBER childMember = findMember(tdec, childReducedMember->member);
//     TU_TDEC_MEMBER parentMember = findMemberParent(tdec, childMember);
//     ReducedMember* parentReducedMember = newcolumn->membersToReducedMembers[parentMember];
// 
//     assert(childReducedMember->type != TYPE_4_CONNECTS_TWO_PATHS);
// 
// #if defined(TU_DEBUG_TDEC)
//     printf("        Merging %s child member %d of type %d with its parent %s member %d of type %d.\n",
//       tdec->members[childMember].type == TDEC_MEMBER_TYPE_BOND ? "bond" :
//       (tdec->members[childMember].type == TDEC_MEMBER_TYPE_POLYGON ? "polygon" : "prime"), childMember,
//       childReducedMember->type, tdec->members[parentMember].type == TDEC_MEMBER_TYPE_BOND ? "bond" :
//       (tdec->members[parentMember].type == TDEC_MEMBER_TYPE_POLYGON ? "polygon" : "prime"), parentMember,
//       parentReducedMember->type);
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     /* Figure out how the parent wants to be connected. Bonds always go via head. */
//     bool parentViaHead = true;
//     TU_TDEC_EDGE markerOfParent = tdec->members[childMember].markerOfParent;
//     TU_TDEC_NODE markerOfParentHead = findEdgeHead(tdec, markerOfParent);
//     TU_TDEC_NODE markerOfParentTail = findEdgeTail(tdec, markerOfParent);
//     if (tdec->members[parentMember].type == TDEC_MEMBER_TYPE_POLYGON)
//     {
//       if (parentReducedMember == reducedComponent->root)
//       {
//         TU_TDEC_NODE pathHead = findEdgeHead(tdec, parentReducedMember->representativePathEdge);
//         TU_TDEC_NODE pathTail = findEdgeTail(tdec, parentReducedMember->representativePathEdge);
//         parentViaHead = pathHead == markerOfParentHead || pathTail == markerOfParentHead;
//       }
//       else
//       {
//         assert(0 == "mergeMembers for non-root parent polygon not implemented.");
//       }
//     }
//     else if (tdec->members[parentMember].type == TDEC_MEMBER_TYPE_PRIME)
//     {
//       assert(0 == "mergeMembers for prime parent not implemented.");
//     }
// 
//     /* Figure out how the child wants to be connected. */
//     TU_TDEC_EDGE markerToParent = tdec->members[childMember].markerToParent;
//     TU_TDEC_NODE markerToParentHead = findEdgeHead(tdec, markerToParent);
//     TU_TDEC_NODE markerToParentTail = findEdgeTail(tdec, markerToParent);
//     bool childViaHead; /* Shall indicate whether this child's subpath ends in parent marker's head node */
// 
//     if (tdec->members[childMember].type == TDEC_MEMBER_TYPE_BOND)
//     {
//       childViaHead = childReducedMember->firstReducedEdge == NULL;
//     }
//     else if (tdec->members[childMember].type == TDEC_MEMBER_TYPE_POLYGON)
//     {
//       if (childReducedMember->type == TYPE_3_EXTENSION)
//       {
//         TU_TDEC_NODE pathHead = findEdgeHead(tdec, childReducedMember->representativePathEdge);
//         TU_TDEC_NODE pathTail = findEdgeTail(tdec, childReducedMember->representativePathEdge);
//         childViaHead = pathHead == markerToParentHead || pathTail == markerToParentHead;
//       }
//       else
//       {
//         assert(0 == "mergeMembers for polygon with type != 3 not implemented.");
//       }
//     }
//     else
//     {
//       assert(0 == "mergeMembers for prime member not implemented.");
//     }
// 
//     /* We first merge the nodes of the two marker edges. */
// 
//     if (parentViaHead == childViaHead)
//     {
//       /* We identify head with head. */
// 
// #if defined(TU_DEBUG_TDEC)
//       printf("          Identifying parent marker head node %d with head node %d.\n",
//         markerToParentHead, markerOfParentHead);
//       printf("          Identifying parent marker tail node %d with tail node %d.\n",
//         markerToParentTail, markerOfParentTail);
//       fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
//       
//       tdec->nodes[markerToParentHead].representativeNode = markerOfParentHead;
//       tdec->nodes[markerToParentTail].representativeNode = markerOfParentTail;
//     }
//     else
//     {
//       /* We identify head with tail. */
//       
// #if defined(TU_DEBUG_TDEC)
//       printf("          Identifying parent marker head node %d with tail node %d.\n",
//         markerToParentHead, markerOfParentTail);
//       printf("          Identifying parent marker tail node %d with head node %d.\n",
//         markerToParentTail, markerOfParentHead);
//       fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//       tdec->nodes[markerToParentHead].representativeNode = markerOfParentTail;
//       tdec->nodes[markerToParentTail].representativeNode = markerOfParentHead;
//     }
// 
//     /* We add the member's edges to the parent's edge list and thereby remove the two marker 
//      * edges. */
//     if (tdec->members[parentMember].firstEdge == markerOfParent)
//       tdec->members[parentMember].firstEdge = tdec->edges[markerOfParent].next;
//     tdec->edges[tdec->edges[markerOfParent].next].prev = tdec->edges[markerToParent].prev;
//     tdec->edges[tdec->edges[markerOfParent].prev].next = tdec->edges[markerToParent].next;
//     tdec->edges[tdec->edges[markerToParent].next].prev = tdec->edges[markerOfParent].prev;
//     tdec->edges[tdec->edges[markerToParent].prev].next = tdec->edges[markerOfParent].next;;
//     tdec->members[parentMember].numEdges += tdec->members[childMember].numEdges - 2;
// 
//     tdec->numEdges -= 2;
//     tdec->edges[markerOfParent].next = tdec->firstFreeEdge;
//     tdec->edges[markerToParent].next = markerOfParent;
//     tdec->firstFreeEdge = markerToParent;
// 
//     /* We now merge the member. */
//     tdec->members[childMember].representativeMember = parentMember;
// 
//     childReducedMember = parentReducedMember;
//   }
// 
//   return TU_OKAY;
// }

// static
// TU_ERROR addColumnUpdate(
//   TU* tu,                             /**< \ref TU environment. */
//   TU_TDEC* tdec,                      /**< t-decomposition. */
//   TU_TDEC_NEWCOLUMN* newcolumn,       /**< new-column structure. */
//   ReducedComponent* reducedComponent, /**< Reduced member. */
//   TU_TDEC_EDGE newEdge                /**< Edge to be added appropriately. */
// )
// {
//   assert(tu);
//   assert(tdec);
//   assert(newcolumn);
// 
// #if defined(TU_DEBUG_TDEC)
//   printf("      Updating reduced decomposition with component %ld, adding edge %d.\n",
//     (reducedComponent - &newcolumn->reducedComponents[0]), newEdge);
//   fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//   if (reducedComponent->terminalMember[0] == reducedComponent->terminalMember[1])
//   {
// #if defined(TU_DEBUG_TDEC)
//     printf("      Unique terminal member %d is %s.\n", reducedComponent->terminalMember[0],
//       tdec->members[reducedComponent->terminalMember[0]].type == TDEC_MEMBER_TYPE_BOND ? "a bond" :
//       (tdec->members[reducedComponent->terminalMember[0]].type == TDEC_MEMBER_TYPE_PRIME ? "prime" :
//       "a polygon"));
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     if (tdec->members[reducedComponent->terminalMember[0]].type == TDEC_MEMBER_TYPE_BOND)
//     {
//       /* Add edge to the bond.  */
// 
//       tdec->edges[newEdge].member = reducedComponent->terminalMember[0];
//       tdec->edges[newEdge].head = -1;
//       tdec->edges[newEdge].tail = -1;
//       TU_CALL( addEdgeToMembersEdgeList(tu, tdec, newEdge, reducedComponent->terminalMember[0]) );
//     }
//     else if (tdec->members[reducedComponent->terminalMember[0]].type == TDEC_MEMBER_TYPE_PRIME)
//     {
//       tdec->edges[newEdge].member = reducedComponent->terminalMember[0];
//       tdec->edges[newEdge].head = reducedComponent->terminalNode[0];
//       tdec->edges[newEdge].tail = reducedComponent->terminalNode[1];
//       TU_CALL( addEdgeToMembersEdgeList(tu, tdec, newEdge, reducedComponent->terminalMember[0]) );
//     }
//     else
//     {
//       assert(0 == "Adding of column with same polygon end members not implemented.");
//     }
//   }
//   else
//   {
// #if defined(TU_DEBUG_TDEC)
//     printf("      Merging components on unique path from nodes %d (member %d) to %d (member %d).\n",
//       reducedComponent->terminalNode[0], reducedComponent->terminalMember[0],
//       reducedComponent->terminalNode[1], reducedComponent->terminalMember[1]);
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     TU_CALL( createMissingNodes(tu, tdec, newcolumn, reducedComponent, reducedComponent->terminalMember[0]) );
//     TU_CALL( createMissingNodes(tu, tdec, newcolumn, reducedComponent, reducedComponent->terminalMember[1]) );
// 
// #if defined(TU_DEBUG_TDEC)
//     printf("      Terminal nodes are now %d and %d.\n", reducedComponent->terminalNode[0],
//       reducedComponent->terminalNode[1]);
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     TU_CALL( mergeMembers(tu, tdec, newcolumn, reducedComponent,
//       newcolumn->membersToReducedMembers[reducedComponent->terminalMember[0]]) );
//     TU_CALL( mergeMembers(tu, tdec, newcolumn, reducedComponent,
//       newcolumn->membersToReducedMembers[reducedComponent->terminalMember[1]]) );
//     tdec->members[findMember(tdec, reducedComponent->root->member)].type = TDEC_MEMBER_TYPE_PRIME;
// 
// #if defined(TU_DEBUG_TDEC)
//     printf("      Adding edge {%d,%d} to merged prime component.\n", findNode(tdec, reducedComponent->terminalNode[0]), 
//       findNode(tdec, reducedComponent->terminalNode[1]));
//     fflush(stdout);
// #endif /* TU_DEBUG_TDEC */
// 
//     /* Add edge to the prime component.  */
//     tdec->edges[newEdge].member = reducedComponent->root->member;
//     tdec->edges[newEdge].head = findNode(tdec, reducedComponent->terminalNode[0]);
//     tdec->edges[newEdge].tail = findNode(tdec, reducedComponent->terminalNode[1]);
//     TU_CALL( addEdgeToMembersEdgeList(tu, tdec, newEdge, reducedComponent->root->member) );
//   }
// 
//   return TU_OKAY;
// }

static
TU_ERROR doReorderComponent(
  TU* tu,                           /**< \ref TU environment. */
  TU_TDEC* tdec,                    /**< t-decomposition. */
  TU_TDEC_MEMBER member,            /**< Member to be processed. */
  TU_TDEC_MEMBER newParent,         /**< New parent of \p member. */
  TU_TDEC_MEMBER newMarkerToParent, /**< New marker edge linked to new parent of \p member. */
  TU_TDEC_MEMBER markerOfNewParent  /**< Counterpart to \p newMarkerToParent. */
)
{
  assert(tu);
  assert(tdec);
  assert(member >= 0);
  assert(newParent >= 0);

  TU_TDEC_MEMBER oldParent = findMemberParent(tdec, member);
  TU_TDEC_EDGE oldMarkerToParent = tdec->members[member].markerToParent;
  TU_TDEC_EDGE oldMarkerOfParent = tdec->members[member].markerOfParent;

  TUdbgMsg(8, "Flipping parenting of member %d with old parent %d and new parent %d.\n", member, oldParent, newParent);
  
  tdec->members[member].markerToParent = newMarkerToParent;
  tdec->members[member].markerOfParent = markerOfNewParent;
  tdec->edges[markerOfNewParent].childMember = member;

  if (oldMarkerToParent >= 0)
    TU_CALL( doReorderComponent(tu, tdec, oldParent, member, oldMarkerOfParent, oldMarkerToParent) );

  return TU_OKAY;
}

static
TU_ERROR reorderComponent(
  TU* tu,                 /**< \ref TU environment. */
  TU_TDEC* tdec,          /**< t-decomposition. */
  TU_TDEC_MEMBER newRoot  /**< The new root of the component. */
)
{
  assert(tu);
  assert(tdec);
  assert(newRoot >= 0 && newRoot < tdec->memMembers);
  assert(isRepresentativeMember(tdec, newRoot));

  TUdbgMsg(4, "Making member %d the new root of its component.\n", newRoot);

  if (tdec->members[newRoot].parentMember >= 0)
  {
    TU_CALL( doReorderComponent(tu, tdec, findMemberParent(tdec, newRoot), newRoot,
      tdec->members[newRoot].markerOfParent, tdec->members[newRoot].markerToParent) );
  }
  
  return TU_OKAY;
}

static
TU_ERROR mergeLeafBonds(
  TU* tu,       /**< \ref TU environment. */
  TU_TDEC* tdec /**< t-decomposition. */
)
{
  assert(tu);
  assert(tdec);

  for (TU_TDEC_MEMBER member = 0; member < tdec->numMembers; ++member)
  {
    if (isRepresentativeMember(tdec, member) && tdec->members[member].type == TDEC_MEMBER_TYPE_BOND
      && tdec->members[member].numEdges == 2 && tdec->members[member].parentMember >= 0)
    {
      TU_TDEC_MEMBER parentMember = tdec->members[member].parentMember;
      TU_TDEC_EDGE parentEdge = tdec->members[member].markerOfParent;
      TU_TDEC_EDGE childEdge = tdec->members[member].markerToParent;
      TU_TDEC_EDGE otherBondEdge = tdec->edges[childEdge].next;
      TUdbgMsg(4, "Merging bond %d into its parent %d.\n", member, parentMember);

      /* We just use the nodes of the parent's child marker (even if -1). */
      tdec->edges[otherBondEdge].head = tdec->edges[parentEdge].head;
      tdec->edges[otherBondEdge].tail = tdec->edges[parentEdge].tail;

      /* Identify members. */
      tdec->members[member].representativeMember = parentMember;

      /* We replace the parent's child marker edge by the members non-parent marker edge and remove the two marker edges. */
      if (tdec->members[parentMember].firstEdge == parentEdge)
        tdec->members[parentMember].firstEdge = tdec->edges[parentEdge].next;
      tdec->edges[tdec->edges[parentEdge].next].prev = otherBondEdge;
      tdec->edges[tdec->edges[parentEdge].prev].next = otherBondEdge;
      tdec->edges[otherBondEdge].prev = tdec->edges[parentEdge].prev;
      tdec->edges[otherBondEdge].next = tdec->edges[parentEdge].next;
      tdec->edges[otherBondEdge].member = findMember(tdec, parentMember);
      tdec->numEdges -= 2;
      tdec->edges[parentEdge].next = tdec->firstFreeEdge;
      tdec->edges[childEdge].next = parentEdge;
      tdec->firstFreeEdge = childEdge;
    }
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

  TUdbgMsg(0, "\n  Adding a column with %d 1's.\n", numEntries);

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  /* Create reduced components for new edges. */
  TU_CALL( completeReducedDecomposition(tu, tdec, newcolumn, entryRows, numEntries) );
  
  /* Process all reduced components individually. */

  TU_TDEC_EDGE* componentNewEdges = NULL;
  TU_CALL( TUallocStackArray(tu, &componentNewEdges, newcolumn->numReducedComponents) );

  int maxDepthComponent = -1;
  TUdbgMsg(4, "Processing %d reduced components.\n", newcolumn->numReducedComponents);
  for (int i = 0; i < newcolumn->numReducedComponents; ++i)
  {
    ReducedComponent* reducedComponent = &newcolumn->reducedComponents[i];

    TUdbgMsg(4, "Processing reduced component %d of depth %d.\n", i, reducedComponent->rootDepth);

    if (maxDepthComponent < 0 || reducedComponent->rootDepth
      > newcolumn->reducedComponents[maxDepthComponent].rootDepth)
    {
      maxDepthComponent = i;
    }

    TU_CALL( addColumnProcessComponent(tu, tdec, newcolumn, reducedComponent, reducedComponent->root, 0) );

    assert(reducedComponent->numTerminals == 2);
    TUdbgMsg(6, "Terminal members are %d and %d.\n", reducedComponent->terminalMember[0],
      reducedComponent->terminalMember[1]);
    TUdbgMsg(6, "Terminal nodes are %d and %d.\n", reducedComponent->terminalNode[0],
      reducedComponent->terminalNode[1]);
    assert(findMember(tdec, reducedComponent->terminalMember[0]) == findMember(tdec, reducedComponent->terminalMember[1]));

    /* Create new edge for this component. If there is one component, this is a column edge, and
     * otherwise it is a marker edge that will be linked to a new polygon consisting of all these
     * marker edges and the column edge. */
    TU_TDEC_EDGE newEdge;
    TU_CALL( createEdge(tu, tdec, -1, &newEdge) );
    componentNewEdges[i] = newEdge;
    tdec->edges[newEdge].childMember = -1;
    tdec->edges[newEdge].member = findMember(tdec, reducedComponent->terminalMember[0]);
    tdec->edges[newEdge].head = reducedComponent->terminalNode[0];
    tdec->edges[newEdge].tail = reducedComponent->terminalNode[1];
    tdec->edges[newEdge].name = INT_MAX/2;
    TU_CALL( addEdgeToMembersEdgeList(tu, tdec, newEdge, tdec->edges[newEdge].member) );
  }

  for (int c = 0; c < newcolumn->numReducedComponents; ++c)
  {
    if (c == maxDepthComponent)
      TUdbgMsg(6, "Reduced component %d has maximum depth and will remain a root.\n", c);
    else
      TU_CALL( reorderComponent(tu, tdec, findMember(tdec, tdec->edges[componentNewEdges[c]].member)) );
  }
  
  if (newcolumn->numReducedComponents == 0)
  {
    assert(0 == "Adding a zero column not implemented.");
  }
  else if (newcolumn->numReducedComponents == 1)
  {
    TU_TDEC_EDGE columnEdge = componentNewEdges[0];
    tdec->edges[columnEdge].name = -1 - column;
    tdec->edges[columnEdge].childMember = -1;
  }
  else
  {
    /*
     * We create another edge for the column as well as a polygon containing all new edges and this one.
     */

    TU_TDEC_MEMBER polygon;
    TU_CALL( createMember(tu, tdec, TDEC_MEMBER_TYPE_POLYGON, &polygon) );

    TU_TDEC_EDGE columnEdge;
    TU_CALL( createEdge(tu, tdec, polygon, &columnEdge) );
    tdec->edges[columnEdge].childMember = -1;
    tdec->edges[columnEdge].head = -1;
    tdec->edges[columnEdge].tail = -1;
    tdec->edges[columnEdge].name = -1 - column;
    TU_CALL( addEdgeToMembersEdgeList(tu, tdec, columnEdge, polygon) );

    for (int i = 0; i < newcolumn->numReducedComponents; ++i)
    {
      TU_TDEC_EDGE newEdge = componentNewEdges[i];
      TU_TDEC_EDGE markerEdge;
      TU_CALL( createEdge(tu, tdec, polygon, &markerEdge) );
      TU_CALL( addEdgeToMembersEdgeList(tu, tdec, markerEdge, polygon) );
      tdec->edges[markerEdge].head = -1;
      tdec->edges[markerEdge].tail = -1;

      TU_TDEC_MEMBER partnerMember = findEdgeMember(tdec, newEdge);

      if (i == maxDepthComponent)
      {
        tdec->edges[markerEdge].childMember = -1;
        tdec->edges[markerEdge].name = INT_MAX - tdec->numMarkers;
        tdec->members[polygon].parentMember = partnerMember;
        tdec->members[polygon].markerToParent = markerEdge;
        tdec->members[polygon].markerOfParent = newEdge;
        tdec->edges[newEdge].name = -INT_MAX + tdec->numMarkers;
        tdec->edges[newEdge].childMember = polygon;
      }
      else
      {
        tdec->edges[markerEdge].childMember = partnerMember;
        tdec->members[partnerMember].markerOfParent = markerEdge;
        tdec->members[partnerMember].markerToParent = newEdge;
        tdec->members[partnerMember].parentMember = polygon;
        tdec->edges[markerEdge].name = INT_MAX - tdec->numMarkers;
        tdec->edges[newEdge].name = -INT_MAX + tdec->numMarkers;
      }

      tdec->numMarkers++;
    }
  }

#if defined(TU_DEBUG_DOT)
  TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
  
  TU_CALL( mergeLeafBonds(tu, tdec) );

#if defined(TU_DEBUG_DOT)
  TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */
  

  TU_CALL( TUfreeStackArray(tu, &componentNewEdges) );

  newcolumn->numReducedMembers = 0;
  newcolumn->numReducedComponents = 0;

  TUconsistencyAssert( TUtdecConsistency(tu, tdec) );

  return TU_OKAY;
}

TU_ERROR testGraphicnessTDecomposition(TU* tu, TU_CHRMAT* matrix, TU_CHRMAT* transpose,
  bool* pisGraphic, TU_GRAPH* graph, TU_GRAPH_EDGE* basis, TU_GRAPH_EDGE* cobasis,
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
    *pisGraphic = true;
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

  TU_TDEC* tdec = NULL;
  TU_CALL( TUtdecCreate(tu, &tdec, 0, 0, 0, 0, 0) ); /* TODO: avoid reallocations. */

  /* Process each column. */
  TU_TDEC_NEWCOLUMN* newcolumn = NULL;
  TUtdecnewcolumnCreate(tu, &newcolumn);
  *pisGraphic = true;
  for (int column = 0; column < matrix->numColumns; ++column)
  {
    TU_CALL( TUtdecAddColumnCheck(tu, tdec, newcolumn,
      &transpose->entryColumns[transpose->rowStarts[column]],
      transpose->rowStarts[column+1] - transpose->rowStarts[column]) );

#if defined(TU_DEBUG_DOT)
    TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

    if (newcolumn->remainsGraphic)
    {
      TU_CALL( TUtdecAddColumnApply(tu, tdec, newcolumn, column, &transpose->entryColumns[transpose->rowStarts[column]],
        transpose->rowStarts[column+1] - transpose->rowStarts[column]) );
    }
    else
    {
      *pisGraphic = false;
      assert(!"Not implemented");
    }
  }

#if defined(TU_DEBUG_DOT)
  TU_CALL( debugDot(tu, tdec, newcolumn) );
#endif /* TU_DEBUG_DOT */

  TU_CALL( TUtdecnewcolumnFree(tu, &newcolumn) );

  if (*pisGraphic && graph)
  {
    TU_CALL( TUtdecToGraph(tu, tdec, graph, true, basis, cobasis, NULL) );
  }

  TU_CALL( TUtdecFree(tu, &tdec) );

  return TU_OKAY;
}

