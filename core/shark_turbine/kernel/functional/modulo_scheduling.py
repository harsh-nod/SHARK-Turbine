#!/usr/bin/env python3
# Modulo Scheduling for Cyclic Graphs
from copy import deepcopy
import numpy as np

disabled_graphviz = False
try:
    import pygraphviz as pgv
except:
    disabled_graphviz = True


class Node:
    def __init__(self, label, RRT) -> None:
        self.label = label
        self.f = 0
        self.RRT = RRT
        self.leader = None


class Edge:
    def __init__(self, label, fromNode, toNode, delay, iterationDelay) -> None:
        self.label = label
        self.fromNode = fromNode
        self.toNode = toNode
        self.delay = delay
        self.iterationDelay = iterationDelay
        self.delta = 0


class Graph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = {}
        self.debug = False

    def addNode(self, node):
        self.nodes.append(node)

    def addEdge(self, edge):
        if edge.fromNode not in self.edges:
            self.edges[edge.fromNode] = []
        self.edges[edge.fromNode].append(edge)

    def generateDotGraph(self):
        if not disabled_graphviz:
            G = pgv.AGraph(directed=True)
            for node in self.nodes:
                G.add_node(node.label)
            for node, edges in self.edges.items():
                for edge in edges:
                    G.add_edge(
                        node.label,
                        edge.toNode.label,
                        label=f"{edge.delay, edge.iterationDelay}",
                    )
            G.draw("dependence_graph.png", prog="dot")

    def runDFSLoop(self, edges, iter):
        self.exploredNodes = []
        self.t = 0
        self.s = None
        self.nodes.sort(key=lambda x: x.f, reverse=True)
        for node in self.nodes:
            if node not in self.exploredNodes:
                self.s = node
                self.runDFS(node, edges, iter)

    def runDFS(self, node, edges, iter):
        self.exploredNodes.append(node)
        node.leader = self.s
        if node in edges:
            for edge in edges[node]:
                if edge.toNode not in self.exploredNodes:
                    self.runDFS(edge.toNode, edges, iter)
        self.t += 1
        if iter == 0:
            node.f = self.t
            if self.debug:
                print("finishing time for " + node.label + " is " + str(node.f))

    def findSCC(self):
        reversedEdges = {}
        # Randomly assign finishing times to nodes
        for i, node in enumerate(self.nodes):
            node.f = i
        # Construct reversed edges
        for fromNode, edges in self.edges.items():
            for edge in edges:
                reversed = deepcopy(edge)
                reversed.fromNode = edge.toNode
                reversed.toNode = fromNode
                if not edge.toNode in reversedEdges:
                    reversedEdges[edge.toNode] = []
                reversedEdges[edge.toNode].append(reversed)
        # Run DFS on reversed graph
        self.runDFSLoop(reversedEdges, 0)
        # Run DFS on graph
        self.runDFSLoop(self.edges, 1)
        self.SCCs = {}
        for node in self.nodes:
            if node.leader not in self.SCCs:
                self.SCCs[node.leader] = []
            self.SCCs[node.leader].append(node)

        for leader, nodes in self.SCCs.items():
            if self.debug:
                print("Leader = ", leader.label)
            for node in nodes:
                if self.debug:
                    print("has node = ", node.label, node.f)


class ModuloScheduler:
    def __init__(self, resourceVector, dependenceGraph: Graph, debug=False) -> None:
        self.resourceVector = resourceVector
        self.dependenceGraph = dependenceGraph
        self.dependenceGraph.debug = debug
        self.debug = debug

    def computeResMII(self):
        usage = [0 for _ in range(len(self.resourceVector))]
        for node in self.dependenceGraph.nodes:
            for edge in node.RRT:
                for i, resource_util in enumerate(edge):
                    usage[i] += resource_util
        usage_per_resource = [x / y for x, y in zip(usage, self.resourceVector)]
        return np.max(usage_per_resource)

    def computeRecMII(self):
        recMII = -1
        for _, nodes in self.dependenceGraph.SCCs.items():
            if len(nodes) == 1:
                continue
            delay = 0
            iterationDelay = 0
            nodes.sort(key=lambda x: x.f)
            for i, node in enumerate(nodes):
                for edge in self.dependenceGraph.edges[node]:
                    nextNode = nodes[0] if i == len(nodes) - 1 else nodes[i + 1]
                    if edge.toNode.label == nextNode.label:
                        delay += edge.delay
                        iterationDelay += edge.iterationDelay
            mii = delay / iterationDelay
            if mii > recMII:
                recMII = mii
        return recMII

    def computeAllPairsLongestPath(self, T):
        self.estar = {}
        for fromNode, edges in self.dependenceGraph.edges.items():
            for edge in edges:
                nodeLabel = (fromNode, edge.toNode)
                self.estar[nodeLabel] = edge
                self.estar[nodeLabel].delta = edge.delay
                if edge.iterationDelay > 0:
                    self.estar[nodeLabel].delta -= T

        for fromNode, edges in self.dependenceGraph.edges.items():
            stack = []
            for edge in edges:
                stack.append(edge)
            while stack:
                startEdge = stack.pop(0)
                middleNode = startEdge.toNode
                for edge in self.dependenceGraph.edges[middleNode]:
                    if edge.toNode == fromNode:
                        continue
                    if (fromNode, edge.toNode) in self.estar:
                        if (
                            self.estar[(fromNode, edge.toNode)].delta
                            >= edge.delta + startEdge.delta
                        ):
                            continue
                    edgeLabel = fromNode.label + "->" + edge.toNode.label
                    newEdge = Edge(edgeLabel, fromNode, edge.toNode, 0, 0)
                    newEdge.delta = edge.delta + startEdge.delta
                    self.estar[(fromNode, edge.toNode)] = newEdge
                    stack.append(newEdge)

        for (fromNode, toNode), edge in self.estar.items():
            if self.debug:
                print(edge.label, edge.delta)

    def SCCScheduled(self, RT, T, c, first, s):
        if self.debug:
            print(f"Scheduling {first.label} at time = ", s)
        RTp = deepcopy(RT)
        if not self.nodeScheduled(RTp, T, first, s):
            return False
        if self.debug:
            print(RTp)
        # TODO: Prioritize the traversal of these nodes
        for node in c:
            if node == first:
                if self.debug:
                    print("Skipping ...", node.label)
                continue
            if self.debug:
                print(f"Scheduling {node.label} ...")
            slmax = 0
            for (fromNode, toNode), edge in self.estar.items():
                if (
                    toNode.label == node.label
                    and fromNode in c
                    and fromNode in self.schedule
                ):
                    sl = self.schedule[fromNode] + edge.delta
                    if slmax < sl:
                        slmax = sl

            sumin = 100
            for (fromNode, toNode), edge in self.estar.items():
                if (
                    fromNode.label == node.label
                    and toNode in c
                    and toNode in self.schedule
                ):
                    su = self.schedule[toNode] - edge.delta
                    if su < sumin:
                        sumin = su

            schedulingSucceeded = False
            if self.debug:
                print("Range = ", slmax, min(sumin, slmax + T - 1))
            for s in range(slmax, min(sumin, slmax + T - 1) + 1):
                if self.debug:
                    print(f"Scheduling {node.label} at time = ", s)
                if self.nodeScheduled(RTp, T, node, s):
                    schedulingSucceeded = True
                    break

            if not schedulingSucceeded:
                if self.debug:
                    print("Scheduling failed!\n")
                return False
        RT[:] = deepcopy(RTp)
        return True

    def nodeScheduled(self, RT, T, n, s):
        RTP = deepcopy(RT)
        for i, row in enumerate(n.RRT):
            RTP[(s + i) % T] = list(np.array(RTP[(s + i) % T]) + np.array(row))
        if self.debug:
            print("Original:", RT)
        if self.debug:
            print("Updated:", RTP)

        validResourceUsage = True
        for row in RTP:
            if np.any(np.array(row) > np.array(self.resourceVector)):
                validResourceUsage = False
                break

        if validResourceUsage:
            RT[:] = deepcopy(RTP)
            self.schedule[n] = s
            return True
        return False

    def allSCCScheduled(self, SCC):
        for nodes in SCC:
            for node in nodes:
                if node not in self.schedule:
                    return False
        return True

    def sortSCCs(self, SCCs):
        times = []
        leaders = []
        for leader, scc in SCCs.items():
            maxTime = -1
            for node in scc:
                if node.f > maxTime:
                    maxTime = node.f
            times.append(maxTime)
            leaders.append(leader)
        keys = [b[0] for b in sorted(zip(leaders, times), key=lambda x: x[1])]
        return [SCCs[i] for i in keys]

    def generateSchedule(self):
        self.dependenceGraph.findSCC()
        self.eprime = []
        for _, edges in self.dependenceGraph.edges.items():
            for edge in edges:
                if edge.iterationDelay == 0:
                    self.eprime.append(edge)
        # Compute starting II
        resMII = self.computeResMII()
        recMII = self.computeRecMII()
        T0 = int(max(recMII, resMII))
        # Start scheduling
        done = False
        T = T0
        Tmax = T0 * 3
        sorted_SCCs = self.sortSCCs(self.dependenceGraph.SCCs)
        for T in range(T0, T0 + Tmax):
            if self.debug:
                print("T = ", T)
            self.schedule = {}
            self.RT = [
                [0 for _ in range(len(self.resourceVector))] for _ in range(int(T))
            ]
            self.computeAllPairsLongestPath(T)
            for scc in sorted_SCCs:
                s0 = {}
                for node in scc:
                    maxS = 0
                    for (fromNode, toNode), edge in self.estar.items():
                        if node.label == toNode.label and fromNode in self.schedule:
                            s = self.schedule[fromNode] + edge.delta
                            if s > maxS:
                                maxS = s
                    s0[node] = maxS
                first = min(s0, key=s0.get)
                s0 = s0[first]
                schedulingSucceeded = False
                for s in range(s0, s0 + T):
                    if self.SCCScheduled(self.RT, T, scc, first, s):
                        schedulingSucceeded = True
                        break
                if not schedulingSucceeded:
                    break
            if self.allSCCScheduled(sorted_SCCs):
                break

        if schedulingSucceeded:
            print("Success!")
        else:
            print("Failed to schedule!")
        print("Schedule = ", [(t, x.label) for x, t in self.schedule.items()])
        print("RRT = ", self.RT)
