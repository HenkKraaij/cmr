// #define TU_DEBUG /* Uncomment to debug graph operations. */
// #define TU_DEBUG_CONSISTENCY /* Uncomment to check consistency of t-decompositions. */

#include <tu/graph.h>

#include "env_internal.h"
#include "hashtable.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>

#define isValid(nodeOrArc) \
  ((nodeOrArc) >= 0)

void TUgraphEnsureConsistent(TU* tu, TU_GRAPH* graph)
{
  assert(tu);
  assert(graph);

  TUdbgMsg(0, "Ensuring consistency of listgraph with %d nodes and %d edges.\n", TUgraphNumNodes(graph),
    TUgraphNumEdges(graph));

  /* Count nodes and check prev/next linked lists. */

  int countNodes = 0;
#if !defined(NDEBUG)
  TU_GRAPH_NODE u = -1;
#endif /* !NDEBUG */
  for (TU_GRAPH_NODE v = TUgraphNodesFirst(graph); TUgraphNodesValid(graph, v); v = TUgraphNodesNext(graph, v))
  {
    assert(graph->nodes[v].prev == u);
#if !defined(NDEBUG)
    u = v;
#endif /* !NDEBUG */
    ++countNodes;
  }
  assert(graph->numNodes == countNodes);

  /* Count free nodes. */

  int countFree = 0;
  TU_GRAPH_NODE v = graph->freeNode;
  while (isValid(v))
  {
    v = graph->nodes[v].next;
    ++countFree;
  }
  assert(countFree + countNodes == graph->memNodes);

  /* Check outgoing and incoming arcs. */

  int countIncident = 0;
  int countLoops = 0;
  for (TU_GRAPH_NODE v = TUgraphNodesFirst(graph); TUgraphNodesValid(graph, v); v = TUgraphNodesNext(graph, v))
  {
    TUdbgMsg(0, "First out-arc of node %d is %d\n", v, graph->nodes[v].firstOut);

    /* Check lists for outgoing arcs. */
    for (TU_GRAPH_ITER i = TUgraphIncFirst(graph, v); TUgraphIncValid(graph, i); i = TUgraphIncNext(graph, i))
    {
      TUdbgMsg(0, "Current arc is %d = (%d,%d), opposite arc is %d, prev is %d, next is %d.\n", i,
        graph->arcs[i ^ 1].target, graph->arcs[i].target, i^1, graph->arcs[i].prev, graph->arcs[i].next);

      if (TUgraphIncSource(graph, i) == TUgraphIncTarget(graph, i))
        ++countLoops;
      assert(graph->arcs[i ^ 1].target == v);
      ++countIncident;
    }
  }
  assert(countIncident + countLoops == 2 * graph->numEdges);

  countFree = 0;
  int e = graph->freeEdge;

  TUdbgMsg(0, "freeEdge = %d\n", e);

  while (isValid(e))
  {
    e = graph->arcs[2*e].next;
    ++countFree;
  }

  assert(countIncident + countLoops + 2 * countFree == 2 * graph->memEdges);

  TUdbgMsg(0, "Consistency checked.\n");
}

TU_ERROR TUgraphCreateEmpty(TU* tu, TU_GRAPH** pgraph, int memNodes, int memEdges)
{
  assert(tu);
  assert(pgraph);
  assert(*pgraph == NULL);

  TU_CALL( TUallocBlock(tu, pgraph) );
  TU_GRAPH* graph = *pgraph;
  graph->numNodes = 0;
  if (memNodes <= 0)
    memNodes = 1;
  graph->memNodes = memNodes;
  graph->nodes = NULL;
  TU_CALL( TUallocBlockArray(tu, &graph->nodes, memNodes) );
  graph->numEdges = 0;
  if (memEdges <= 0)
    memEdges = 1;
  graph->memEdges = memEdges;
  graph->arcs = NULL;
  TU_CALL( TUallocBlockArray(tu, &graph->arcs, 2 * memEdges) );
  graph->firstNode = -1;
  graph->freeNode = (memNodes > 0) ? 0 : -1;
  for (int v = 0; v < graph->memNodes - 1; ++v)
    graph->nodes[v].next = v+1;
  graph->nodes[graph->memNodes-1].next = -1;
  graph->freeEdge = (memEdges > 0) ? 0 : -1;
  for (int e = 0; e < graph->memEdges - 1; ++e)
    graph->arcs[2*e].next = e+1;
  graph->arcs[2*graph->memEdges-2].next = -1;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  return TU_OKAY;
}

TU_ERROR TUgraphFree(TU* tu, TU_GRAPH** pgraph)
{
  assert(pgraph);

  TU_GRAPH* graph = *pgraph;
  if (!graph)
    return TU_OKAY;

  TUdbgMsg(0, "TUgraphFree(|V|=%d, |E|=%d)\n", TUgraphNumNodes(graph), TUgraphNumEdges(graph));

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  TU_CALL( TUfreeBlockArray(tu, &graph->nodes) );
  TU_CALL( TUfreeBlockArray(tu, &graph->arcs) );

  TU_CALL( TUfreeBlock(tu, pgraph) );
  *pgraph = NULL;

  return TU_OKAY;
}

TU_ERROR TUgraphClear(TU* tu, TU_GRAPH* graph)
{
  assert(tu);
  assert(graph);

  graph->numNodes = 0;
  graph->numEdges = 0;
  graph->firstNode = -1;
  graph->freeNode = (graph->memNodes > 0) ? 0 : -1;
  for (int v = 0; v < graph->memNodes - 1; ++v)
    graph->nodes[v].next = v+1;
  graph->nodes[graph->memNodes-1].next = -1;
  graph->freeEdge = (graph->memEdges > 0) ? 0 : -1;
  for (int e = 0; e < graph->memEdges - 1; ++e)
    graph->arcs[2*e].next = e+1;
  graph->arcs[2*graph->memEdges-2].next = -1;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  return TU_OKAY;
}

TU_ERROR TUgraphAddNode(TU* tu, TU_GRAPH *graph, TU_GRAPH_NODE* pnode)
{
  assert(tu);
  assert(graph);

  TUdbgMsg(0, "TUgraphAddNode().\n");

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  /* If the free list is empty, we have reallocate. */
  if (!isValid(graph->freeNode))
  {
    int mem = (graph->memNodes < 256 ? 0 : 256) + 2 * graph->memNodes;
    TU_CALL( TUreallocBlockArray(tu, &graph->nodes, mem) );
    assert(graph->nodes);
    for (int v = graph->memNodes; v < mem-1; ++v)
      graph->nodes[v].next = v+1;
    graph->nodes[mem-1].next = -1;
    graph->freeNode = graph->memNodes;
    graph->memNodes = mem;
  }

  /* Add to list. */

  TU_GRAPH_NODE node = graph->freeNode;
  graph->freeNode = graph->nodes[node].next;
  graph->numNodes++;
  graph->nodes[node].firstOut = -1;
  graph->nodes[node].prev = -1;
  graph->nodes[node].next = graph->firstNode;
  if (isValid(graph->firstNode))
    graph->nodes[graph->firstNode].prev = node;
  graph->firstNode = node;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  if (pnode)
    *pnode = node;

  return TU_OKAY;
}

TU_ERROR TUgraphAddEdge(TU* tu, TU_GRAPH* graph, TU_GRAPH_NODE u, TU_GRAPH_NODE v,
  TU_GRAPH_EDGE* pedge)
{
  TUdbgMsg(0, "TUgraphAddEdge(%d,%d).\n", u, v);

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */
  assert(u >= 0);
  assert(u < graph->numNodes);
  assert(v >= 0);
  assert(v < graph->numNodes);

  /* If the free list is empty, we have reallocate. */

  if (!isValid(graph->freeEdge))
  {
    int newMemEdges = (graph->memEdges < 1024 ? 0 : 1024) + 2 * graph->memEdges;
    TU_CALL( TUreallocBlockArray(tu, &graph->arcs, 2*newMemEdges) );
    assert(graph->arcs);
    for (int e = graph->memEdges; e < newMemEdges-1; ++e)
      graph->arcs[2*e].next = (e+1);
    graph->arcs[2*newMemEdges-2].next = -1;
    graph->freeEdge = graph->memEdges;
    graph->memEdges = newMemEdges;
  }

  /* Add to list. */

  TU_GRAPH_EDGE edge = graph->freeEdge;
  int arc = 2*edge;
  graph->freeEdge = graph->arcs[arc].next;
  graph->numEdges++;

  /* Add arc to list of outgoing arcs from u. */

  graph->arcs[arc].target = v;
  int firstOut = graph->nodes[u].firstOut;
  graph->arcs[arc].prev = -1;
  graph->arcs[arc].next = firstOut;
  if (isValid(firstOut))
    graph->arcs[firstOut].prev = arc;
  graph->nodes[u].firstOut = arc;

  ++arc;

  /* Add arc to list of incoming arcs to v. */
  graph->arcs[arc].target = u;
  firstOut = graph->nodes[v].firstOut;
  graph->arcs[arc].prev = -1;
  graph->arcs[arc].next = firstOut;
  if (isValid(firstOut))
    graph->arcs[firstOut].prev = arc;
  graph->nodes[v].firstOut = arc;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  if (pedge)
    *pedge = edge;

  return TU_OKAY;
}

TU_ERROR TUgraphDeleteNode(TU* tu, TU_GRAPH* graph, TU_GRAPH_NODE v)
{
  TUdbgMsg(0, "TUgraphDeleteNode(|V|=%d, |E|=%d, v=%d)\n", TUgraphNumNodes(graph), TUgraphNumEdges(graph), v);

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  /* Remove incident edges of which v is the source. */
  while (isValid(graph->nodes[v].firstOut))
    TUgraphDeleteEdge(tu, graph, graph->nodes[v].firstOut/2);

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  /* Remove node from node list. */
  TU_GRAPH_NODE prev = graph->nodes[v].prev;
  TU_GRAPH_NODE next = graph->nodes[v].next;
  if (isValid(prev))
    graph->nodes[prev].next = next;
  else
    graph->firstNode = next;
  if (isValid(next))
    graph->nodes[next].prev = prev;

  graph->nodes[v].next = graph->freeNode;
  graph->freeNode = v;
  graph->numNodes--;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  return TU_OKAY;
}

TU_ERROR TUgraphDeleteEdge(TU* tu, TU_GRAPH* graph, TU_GRAPH_EDGE e)
{
  TUdbgMsg(0, "TUgraphDeleteEdge(|V|=%d, |E|=%d, %d", TUgraphNumNodes(graph), TUgraphNumEdges(graph), e);

  assert(isValid(e));

  int arc = 2*e;
  TU_GRAPH_NODE u = graph->arcs[arc+1].target;
  TU_GRAPH_NODE v = graph->arcs[arc].target;

  TUdbgMsg(0, " = {%d,%d})\n", u, v);

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  /* Remove from u's list of outgoing arcs. */
  TU_GRAPH_EDGE prev = graph->arcs[arc].prev;
  TU_GRAPH_EDGE next = graph->arcs[arc].next;
  if (isValid(prev))
    graph->arcs[prev].next = next;
  else
    graph->nodes[u].firstOut = next;
  if (isValid(next))
    graph->arcs[next].prev = prev;

  /* Add to free list. */
  graph->arcs[arc].next = graph->freeEdge;
  graph->freeEdge = e;
  graph->numEdges--;

  /* Remove from v's list of outgoing arcs. */
  ++arc;
  prev = graph->arcs[arc].prev;
  next = graph->arcs[arc].next;
  if (isValid(prev))
    graph->arcs[prev].next = next;
  else
    graph->nodes[v].firstOut = next;
  if (isValid(next))
    graph->arcs[next].prev = prev;

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  return TU_OKAY;
}


TU_ERROR TUgraphPrint(FILE* stream, TU_GRAPH* graph)
{
  assert(stream);
  assert(graph);

  printf("Graph with %d nodes and %d edges.\n", TUgraphNumNodes(graph), TUgraphNumEdges(graph));
  for (TU_GRAPH_NODE v = TUgraphNodesFirst(graph); TUgraphNodesValid(graph, v); v = TUgraphNodesNext(graph, v))
  {
    fprintf(stream, "Node %d:\n", v);
    for (TU_GRAPH_ITER i = TUgraphIncFirst(graph, v); TUgraphIncValid(graph, i); i = TUgraphIncNext(graph, i))
    {
      fprintf(stream, "  Edge %d: {%d,%d} {arc = %d}\n", TUgraphIncEdge(graph, i), TUgraphIncSource(graph, i),
        TUgraphIncTarget(graph, i), i);
    }
  }

  return TU_OKAY;
}

TU_ERROR TUgraphMergeNodes(TU* tu, TU_GRAPH* graph, TU_GRAPH_NODE u, TU_GRAPH_NODE v)
{
  assert(graph);
  assert(u >= 0);
  assert(u < graph->memNodes);
  assert(v >= 0);
  assert(v < graph->memNodes);
  assert(u != v);

  int a;
  while ((a = graph->nodes[v].firstOut) >= 0)
  {
    graph->nodes[v].firstOut = graph->arcs[a].next;
    graph->arcs[a ^ 1].target = u;
    graph->arcs[a].next = graph->nodes[u].firstOut;
    if (graph->nodes[u].firstOut >= 0)
      graph->arcs[graph->nodes[u].firstOut].prev = a;
    graph->arcs[a].prev = -1;
    graph->nodes[u].firstOut = a;
  }

#if defined(TU_DEBUG_CONSISTENCY)
  TUgraphEnsureConsistent(tu, graph);
#endif /* TU_DEBUG_CONSISTENCY */

  return TU_OKAY;
}

TU_ERROR TUgraphCreateFromEdgeList(TU* tu, TU_GRAPH** pgraph, TU_ELEMENT** pedgeElements, char*** pnodeLabels,
  FILE* stream)
{
  assert(tu);
  assert(pgraph);
  assert(!*pgraph);
  assert(!pedgeElements || !*pedgeElements);
  assert(!pnodeLabels || !*pnodeLabels);
  assert(stream);

  char* line = NULL;
  size_t length = 0;
  ssize_t numRead;

  char* uToken = NULL;
  char* vToken = NULL;
  char* elementToken = NULL;

  TU_CALL( TUgraphCreateEmpty(tu, pgraph, 256, 1024) );
  TU_GRAPH* graph = *pgraph;
  size_t memNodeLabels = 256;
  if (pnodeLabels)
    TU_CALL( TUallocBlockArray(tu, pnodeLabels, memNodeLabels) );
  size_t memEdgeElements = 256;
  if (pedgeElements)
    TU_CALL( TUallocBlockArray(tu, pedgeElements, memEdgeElements) );

  TU_LINEARHASHTABLE_ARRAY* nodeNames = NULL;
  TU_CALL( TUlinearhashtableArrayCreate(tu, &nodeNames, 8, 1024) );
  
  while ((numRead = getline(&line, &length, stream)) != -1)
  {
    char* s = line;

    /* Scan for whitespace */
    while (*s && isspace(*s))
      ++s;
    if (!*s)
      break;

    /* Scan name of node u. */
    uToken = s;
    while (*s && !isspace(*s))
      ++s;
    *s = '\0';
    ++s;

    /* Scan for whitespace */
    while (*s && isspace(*s))
      ++s;
    if (!*s)
      break;

    /* Scan name of node v. */
    vToken = s;
    while (*s && !isspace(*s))
      ++s;
    *s = '\0';
    ++s;

    /* Scan for whitespace */
    while (*s && isspace(*s))
      ++s;
    if (*s)
    {
      /* Scan element. */
      elementToken = s;
      while (*s && !isspace(*s))
        ++s;
      *s = '\0';
      ++s;
    }
    else
    {
      elementToken = NULL;
    }

    /* Figure out node u if it exists. */

    TU_LINEARHASHTABLE_BUCKET bucket;
    TU_LINEARHASHTABLE_HASH hash;
    TU_GRAPH_NODE uNode; 
    if (TUlinearhashtableArrayFind(nodeNames, uToken, strlen(uToken), &bucket, &hash))
      uNode = (TU_GRAPH_NODE) (size_t) TUlinearhashtableArrayValue(nodeNames, bucket);
    else
    {
      TU_CALL( TUgraphAddNode(tu, graph, &uNode) );
      TU_CALL( TUlinearhashtableArrayInsertBucketHash(tu, nodeNames, uToken, strlen(uToken), bucket, hash,
        (void*) (size_t) uNode) );

      /* Add node label. */
      if (pnodeLabels)
      {
        if (uNode >= memNodeLabels)
        {
          memNodeLabels *= 2;
          TU_CALL( TUreallocBlockArray(tu, pnodeLabels, memNodeLabels) );
        }

        (*pnodeLabels)[uNode] = strdup(uToken);
      }
    }

    /* Figure out node v if it exists. */

    TU_GRAPH_NODE vNode; 
    if (TUlinearhashtableArrayFind(nodeNames, vToken, strlen(vToken), &bucket, &hash))
      vNode = (TU_GRAPH_NODE) (size_t) TUlinearhashtableArrayValue(nodeNames, bucket);
    else
    {
      TU_CALL( TUgraphAddNode(tu, graph, &vNode) );
      TU_CALL( TUlinearhashtableArrayInsertBucketHash(tu, nodeNames, vToken, strlen(vToken), bucket, hash,
        (void*) (size_t) vNode) );
      
      /* Add node label. */
      if (pnodeLabels)
      {
        if (vNode >= memNodeLabels)
        {
          memNodeLabels *= 2;
          TU_CALL( TUreallocBlockArray(tu, pnodeLabels, memNodeLabels) );
        }

        (*pnodeLabels)[vNode] = strdup(vToken);
      }
    }

    /* Extract element. */

    TU_ELEMENT element = 0;
    if (elementToken)
    {
      if (strchr("rRtT-", elementToken[0]))
      {
        sscanf(&elementToken[1], "%d", &element);
        element = -element;
      }
      else if (strchr("cC", elementToken[0]))
        sscanf(&elementToken[1], "%d", &element);
      else
        sscanf(elementToken, "%d", &element);
    }

    TU_GRAPH_EDGE edge;
    TU_CALL( TUgraphAddEdge(tu, graph, uNode, vNode, &edge) );

    if (pedgeElements)
    {
      if (edge >= memEdgeElements)
      {
        do
        {
          memEdgeElements *= 2;
        }
        while (edge >= memEdgeElements);
        TU_CALL( TUreallocBlockArray(tu, pedgeElements, memEdgeElements) );
      }

      (*pedgeElements)[edge] = element;
    }
  }

  TU_CALL( TUlinearhashtableArrayFree(tu, &nodeNames) );
  free(line);

  return TU_OKAY;
}
