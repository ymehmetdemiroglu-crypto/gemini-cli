/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
 * Simplified implementation for persona clustering
 */

import type { Cluster, DataPoint } from '../types.js';

/**
 * HDBSCAN clustering algorithm
 * Finds clusters of varying densities and identifies noise points
 */
export class HDBSCAN {
  private minClusterSize: number;
  private minSamples: number;
  private clusterSelectionEpsilon: number;

  constructor(
    minClusterSize = 5,
    minSamples?: number,
    clusterSelectionEpsilon = 0.0
  ) {
    this.minClusterSize = minClusterSize;
    this.minSamples = minSamples ?? minClusterSize;
    this.clusterSelectionEpsilon = clusterSelectionEpsilon;
  }

  /**
   * Fit the model and return cluster assignments (-1 for noise)
   */
  fit(embeddings: number[][]): number[] {
    const n = embeddings.length;
    if (n < this.minClusterSize) {
      return new Array(n).fill(-1);
    }

    // Step 1: Compute core distances
    const coreDistances = this.computeCoreDistances(embeddings);

    // Step 2: Build mutual reachability graph
    const mutualReachability = this.buildMutualReachabilityGraph(
      embeddings,
      coreDistances
    );

    // Step 3: Build minimum spanning tree
    const mst = this.buildMST(mutualReachability, n);

    // Step 4: Build cluster hierarchy
    const hierarchy = this.buildHierarchy(mst, n);

    // Step 5: Extract clusters using stability
    const assignments = this.extractClusters(hierarchy, n);

    return assignments;
  }

  /**
   * Compute core distance for each point
   * Core distance is the distance to the k-th nearest neighbor
   */
  private computeCoreDistances(embeddings: number[][]): number[] {
    const n = embeddings.length;
    const coreDistances = new Array(n);

    for (let i = 0; i < n; i++) {
      const distances: number[] = [];
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          distances.push(this.euclideanDistance(embeddings[i], embeddings[j]));
        }
      }
      distances.sort((a, b) => a - b);
      coreDistances[i] = distances[Math.min(this.minSamples - 1, distances.length - 1)] || 0;
    }

    return coreDistances;
  }

  /**
   * Build mutual reachability distance matrix
   */
  private buildMutualReachabilityGraph(
    embeddings: number[][],
    coreDistances: number[]
  ): number[][] {
    const n = embeddings.length;
    const graph: number[][] = [];

    for (let i = 0; i < n; i++) {
      graph[i] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          graph[i][j] = 0;
        } else {
          const dist = this.euclideanDistance(embeddings[i], embeddings[j]);
          graph[i][j] = Math.max(coreDistances[i], coreDistances[j], dist);
        }
      }
    }

    return graph;
  }

  /**
   * Build minimum spanning tree using Prim's algorithm
   */
  private buildMST(
    graph: number[][],
    n: number
  ): Array<{ from: number; to: number; weight: number }> {
    const mst: Array<{ from: number; to: number; weight: number }> = [];
    const inMST = new Array(n).fill(false);
    const key = new Array(n).fill(Infinity);
    const parent = new Array(n).fill(-1);

    key[0] = 0;

    for (let count = 0; count < n - 1; count++) {
      // Find minimum key vertex not in MST
      let minKey = Infinity;
      let u = -1;
      for (let v = 0; v < n; v++) {
        if (!inMST[v] && key[v] < minKey) {
          minKey = key[v];
          u = v;
        }
      }

      if (u === -1) break;
      inMST[u] = true;

      if (parent[u] !== -1) {
        mst.push({ from: parent[u], to: u, weight: graph[parent[u]][u] });
      }

      // Update key values
      for (let v = 0; v < n; v++) {
        if (!inMST[v] && graph[u][v] < key[v]) {
          key[v] = graph[u][v];
          parent[v] = u;
        }
      }
    }

    return mst;
  }

  /**
   * Build cluster hierarchy from MST
   */
  private buildHierarchy(
    mst: Array<{ from: number; to: number; weight: number }>,
    n: number
  ): Array<{ nodeA: number; nodeB: number; distance: number; size: number }> {
    // Sort MST edges by weight
    const sortedEdges = [...mst].sort((a, b) => a.weight - b.weight);

    // Union-Find structure
    const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);
    const size = new Array(2 * n - 1).fill(1);
    const hierarchy: Array<{
      nodeA: number;
      nodeB: number;
      distance: number;
      size: number;
    }> = [];

    let nextNode = n;

    const find = (x: number): number => {
      if (parent[x] !== x) {
        parent[x] = find(parent[x]);
      }
      return parent[x];
    };

    const union = (a: number, b: number, distance: number): void => {
      const rootA = find(a);
      const rootB = find(b);

      if (rootA !== rootB) {
        hierarchy.push({
          nodeA: rootA,
          nodeB: rootB,
          distance,
          size: size[rootA] + size[rootB],
        });

        parent[rootA] = nextNode;
        parent[rootB] = nextNode;
        size[nextNode] = size[rootA] + size[rootB];
        nextNode++;
      }
    };

    for (const edge of sortedEdges) {
      union(edge.from, edge.to, edge.weight);
    }

    return hierarchy;
  }

  /**
   * Extract clusters from hierarchy using stability measure
   */
  private extractClusters(
    hierarchy: Array<{
      nodeA: number;
      nodeB: number;
      distance: number;
      size: number;
    }>,
    n: number
  ): number[] {
    if (hierarchy.length === 0) {
      return new Array(n).fill(-1);
    }

    // Simple extraction: use density threshold
    const assignments = new Array(n).fill(-1);
    const unionFind = Array.from({ length: n }, (_, i) => i);

    const find = (x: number): number => {
      if (unionFind[x] !== x) {
        unionFind[x] = find(unionFind[x]);
      }
      return unionFind[x];
    };

    // Find distance threshold based on cluster selection epsilon
    const maxDistance = hierarchy.length > 0
      ? Math.max(...hierarchy.map(h => h.distance))
      : 0;
    const threshold = this.clusterSelectionEpsilon > 0
      ? this.clusterSelectionEpsilon
      : maxDistance * 0.5;

    // Build clusters by merging below threshold
    for (const merge of hierarchy) {
      if (merge.distance <= threshold) {
        // Track roots for union-find (used implicitly through find())
        find(merge.nodeA < n ? merge.nodeA : 0);
        find(merge.nodeB < n ? merge.nodeB : 0);
        if (merge.nodeA < n && merge.nodeB < n) {
          unionFind[merge.nodeA] = Math.min(merge.nodeA, merge.nodeB);
          unionFind[merge.nodeB] = Math.min(merge.nodeA, merge.nodeB);
        }
      }
    }

    // Assign cluster labels
    const clusterRoots = new Map<number, number>();
    let nextCluster = 0;

    for (let i = 0; i < n; i++) {
      const root = find(i);
      if (!clusterRoots.has(root)) {
        clusterRoots.set(root, nextCluster++);
      }
      assignments[i] = clusterRoots.get(root)!;
    }

    // Mark small clusters as noise
    const clusterSizes = new Map<number, number>();
    for (const cluster of assignments) {
      clusterSizes.set(cluster, (clusterSizes.get(cluster) || 0) + 1);
    }

    for (let i = 0; i < n; i++) {
      if ((clusterSizes.get(assignments[i]) || 0) < this.minClusterSize) {
        assignments[i] = -1;
      }
    }

    // Renumber clusters to be contiguous
    const clusterMap = new Map<number, number>();
    let newCluster = 0;
    for (let i = 0; i < n; i++) {
      if (assignments[i] !== -1) {
        if (!clusterMap.has(assignments[i])) {
          clusterMap.set(assignments[i], newCluster++);
        }
        assignments[i] = clusterMap.get(assignments[i])!;
      }
    }

    return assignments;
  }

  /**
   * Calculate Euclidean distance
   */
  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - (b[i] || 0);
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
}

/**
 * Build Cluster objects from HDBSCAN results
 */
export function buildHDBSCANClusters(
  dataPoints: DataPoint[],
  assignments: number[]
): Cluster[] {
  const clusterMap = new Map<number, { members: string[]; embeddings: number[][] }>();

  // Group data points by cluster (ignore noise points with assignment -1)
  for (let i = 0; i < assignments.length; i++) {
    const clusterId = assignments[i];
    if (clusterId === -1) continue; // Skip noise

    if (!clusterMap.has(clusterId)) {
      clusterMap.set(clusterId, { members: [], embeddings: [] });
    }
    const cluster = clusterMap.get(clusterId)!;
    cluster.members.push(dataPoints[i].id);
    cluster.embeddings.push(dataPoints[i].embedding);
  }

  // Build cluster objects with calculated centroids
  const clusters: Cluster[] = [];
  for (const [id, data] of clusterMap) {
    const centroid = calculateCentroid(data.embeddings);
    clusters.push({
      id: `cluster_${id}`,
      centroid,
      members: data.members,
      size: data.members.length,
    });
  }

  return clusters;
}

/**
 * Calculate centroid of a set of embeddings
 */
function calculateCentroid(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return [];

  const dimensions = embeddings[0].length;
  const centroid = new Array(dimensions).fill(0);

  for (const embedding of embeddings) {
    for (let i = 0; i < dimensions; i++) {
      centroid[i] += embedding[i];
    }
  }

  for (let i = 0; i < dimensions; i++) {
    centroid[i] /= embeddings.length;
  }

  return centroid;
}
