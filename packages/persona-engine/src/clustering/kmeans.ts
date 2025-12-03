/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * K-Means clustering implementation with k-means++ initialization
 */

import type { Cluster, DataPoint } from '../types.js';

/**
 * K-Means clustering algorithm with k-means++ initialization
 */
export class KMeans {
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private centroids: number[][] = [];

  constructor(k: number, maxIterations = 300, tolerance = 1e-4) {
    this.k = k;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  /**
   * Fit the model to data and return cluster assignments
   */
  fit(embeddings: number[][]): number[] {
    if (embeddings.length === 0) return [];
    if (embeddings.length < this.k) {
      // Fewer points than clusters - each point is its own cluster
      return embeddings.map((_, i) => i);
    }

    // K-means++ initialization
    this.centroids = this.initializeCentroids(embeddings);

    let assignments = new Array(embeddings.length).fill(-1);
    let prevAssignments: number[];

    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      prevAssignments = [...assignments];

      // Assignment step
      assignments = this.assignClusters(embeddings);

      // Update step
      this.updateCentroids(embeddings, assignments);

      // Check convergence
      if (this.hasConverged(assignments, prevAssignments)) {
        break;
      }
    }

    return assignments;
  }

  /**
   * K-means++ initialization for better starting centroids
   */
  private initializeCentroids(embeddings: number[][]): number[][] {
    const centroids: number[][] = [];
    const n = embeddings.length;

    // Choose first centroid randomly
    const firstIndex = Math.floor(Math.random() * n);
    centroids.push([...embeddings[firstIndex]]);

    // Choose remaining centroids with probability proportional to D(x)²
    for (let i = 1; i < this.k; i++) {
      const distances = embeddings.map((point) =>
        Math.min(...centroids.map((c) => this.squaredDistance(point, c)))
      );

      const totalDistance = distances.reduce((sum, d) => sum + d, 0);
      let cumulative = 0;
      const threshold = Math.random() * totalDistance;

      for (let j = 0; j < n; j++) {
        cumulative += distances[j];
        if (cumulative >= threshold) {
          centroids.push([...embeddings[j]]);
          break;
        }
      }

      // Fallback if no centroid was selected
      if (centroids.length <= i) {
        const randomIndex = Math.floor(Math.random() * n);
        centroids.push([...embeddings[randomIndex]]);
      }
    }

    return centroids;
  }

  /**
   * Assign each point to the nearest centroid
   */
  private assignClusters(embeddings: number[][]): number[] {
    return embeddings.map((point) => {
      let minDist = Infinity;
      let minCluster = 0;

      for (let i = 0; i < this.centroids.length; i++) {
        const dist = this.squaredDistance(point, this.centroids[i]);
        if (dist < minDist) {
          minDist = dist;
          minCluster = i;
        }
      }

      return minCluster;
    });
  }

  /**
   * Update centroids to be the mean of assigned points
   */
  private updateCentroids(embeddings: number[][], assignments: number[]): void {
    const dimensions = embeddings[0]?.length || 0;
    const newCentroids: number[][] = [];
    const counts: number[] = new Array(this.k).fill(0);

    // Initialize new centroids
    for (let i = 0; i < this.k; i++) {
      newCentroids.push(new Array(dimensions).fill(0));
    }

    // Sum points for each cluster
    for (let i = 0; i < embeddings.length; i++) {
      const cluster = assignments[i];
      counts[cluster]++;
      for (let d = 0; d < dimensions; d++) {
        newCentroids[cluster][d] += embeddings[i][d];
      }
    }

    // Calculate means
    for (let i = 0; i < this.k; i++) {
      if (counts[i] > 0) {
        for (let d = 0; d < dimensions; d++) {
          newCentroids[i][d] /= counts[i];
        }
        this.centroids[i] = newCentroids[i];
      }
      // If a cluster has no points, keep the old centroid
    }
  }

  /**
   * Check if clustering has converged
   */
  private hasConverged(current: number[], previous: number[]): boolean {
    if (current.length !== previous.length) return false;

    let changes = 0;
    for (let i = 0; i < current.length; i++) {
      if (current[i] !== previous[i]) changes++;
    }

    return changes / current.length < this.tolerance;
  }

  /**
   * Calculate squared Euclidean distance
   */
  private squaredDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - (b[i] || 0);
      sum += diff * diff;
    }
    return sum;
  }

  /**
   * Get the fitted centroids
   */
  getCentroids(): number[][] {
    return this.centroids;
  }

  /**
   * Calculate inertia (sum of squared distances to nearest centroid)
   */
  calculateInertia(embeddings: number[][], assignments: number[]): number {
    let inertia = 0;
    for (let i = 0; i < embeddings.length; i++) {
      const centroid = this.centroids[assignments[i]];
      inertia += this.squaredDistance(embeddings[i], centroid);
    }
    return inertia;
  }
}

/**
 * Build Cluster objects from clustering results
 */
export function buildClusters(
  dataPoints: DataPoint[],
  assignments: number[],
  centroids: number[][]
): Cluster[] {
  const clusterMap = new Map<number, string[]>();

  // Group data points by cluster
  for (let i = 0; i < assignments.length; i++) {
    const clusterId = assignments[i];
    if (!clusterMap.has(clusterId)) {
      clusterMap.set(clusterId, []);
    }
    clusterMap.get(clusterId)!.push(dataPoints[i].id);
  }

  // Build cluster objects
  const clusters: Cluster[] = [];
  for (const [id, members] of clusterMap) {
    clusters.push({
      id: `cluster_${id}`,
      centroid: centroids[id] || [],
      members,
      size: members.length,
    });
  }

  return clusters.filter((c) => c.size > 0);
}

/**
 * Find optimal k using elbow method with inertia
 */
export function findOptimalK(
  embeddings: number[][],
  minK = 2,
  maxK = 10
): number {
  if (embeddings.length < minK) return 1;

  const maxClusters = Math.min(maxK, embeddings.length);
  const inertias: number[] = [];

  for (let k = minK; k <= maxClusters; k++) {
    const kmeans = new KMeans(k);
    const assignments = kmeans.fit(embeddings);
    inertias.push(kmeans.calculateInertia(embeddings, assignments));
  }

  // Find elbow using second derivative
  let maxCurvature = 0;
  let optimalK = minK;

  for (let i = 1; i < inertias.length - 1; i++) {
    // Second derivative (discrete approximation)
    const curvature = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1];
    if (curvature > maxCurvature) {
      maxCurvature = curvature;
      optimalK = minK + i;
    }
  }

  return optimalK;
}
