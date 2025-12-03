/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Cluster Validity Indices (CVI)
 * Statistical metrics to validate cluster quality
 */

import type { Cluster, ClusterValidityIndices, DataPoint } from '../types.js';

/**
 * Calculate all cluster validity indices
 */
export function calculateClusterValidityIndices(
  dataPoints: DataPoint[],
  clusters: Cluster[]
): ClusterValidityIndices {
  if (clusters.length === 0 || dataPoints.length === 0) {
    return {
      silhouette: 0,
      daviesBouldin: Infinity,
      calinskiHarabasz: 0,
      dunn: 0,
    };
  }

  // Build lookup maps
  const pointMap = new Map<string, DataPoint>();
  for (const point of dataPoints) {
    pointMap.set(point.id, point);
  }

  // Get assignments
  const assignments = new Map<string, number>();
  for (let i = 0; i < clusters.length; i++) {
    for (const memberId of clusters[i].members) {
      assignments.set(memberId, i);
    }
  }

  // Calculate indices
  return {
    silhouette: calculateSilhouetteScore(pointMap, clusters, assignments),
    daviesBouldin: calculateDaviesBouldinIndex(pointMap, clusters, assignments),
    calinskiHarabasz: calculateCalinskiHarabaszIndex(pointMap, clusters, assignments),
    dunn: calculateDunnIndex(pointMap, clusters, assignments),
  };
}

/**
 * Silhouette Score
 * Measures how similar a point is to its own cluster vs other clusters
 * Range: -1 to 1 (higher is better)
 */
export function calculateSilhouetteScore(
  pointMap: Map<string, DataPoint>,
  clusters: Cluster[],
  assignments: Map<string, number>
): number {
  if (clusters.length < 2) return 0;

  const silhouettes: number[] = [];

  for (const [pointId, clusterIdx] of assignments) {
    const point = pointMap.get(pointId);
    if (!point) continue;

    // Calculate a(i): average distance to points in same cluster
    const sameCluster = clusters[clusterIdx];
    let sumSameCluster = 0;
    let countSame = 0;

    for (const otherId of sameCluster.members) {
      if (otherId === pointId) continue;
      const other = pointMap.get(otherId);
      if (other) {
        sumSameCluster += euclideanDistance(point.embedding, other.embedding);
        countSame++;
      }
    }

    const a = countSame > 0 ? sumSameCluster / countSame : 0;

    // Calculate b(i): minimum average distance to points in other clusters
    let b = Infinity;

    for (let i = 0; i < clusters.length; i++) {
      if (i === clusterIdx) continue;

      let sumOtherCluster = 0;
      let countOther = 0;

      for (const otherId of clusters[i].members) {
        const other = pointMap.get(otherId);
        if (other) {
          sumOtherCluster += euclideanDistance(point.embedding, other.embedding);
          countOther++;
        }
      }

      if (countOther > 0) {
        b = Math.min(b, sumOtherCluster / countOther);
      }
    }

    // Silhouette coefficient for this point
    if (b === Infinity) b = 0;
    const maxAB = Math.max(a, b);
    const silhouette = maxAB > 0 ? (b - a) / maxAB : 0;
    silhouettes.push(silhouette);
  }

  // Return average silhouette
  return silhouettes.length > 0
    ? silhouettes.reduce((sum, s) => sum + s, 0) / silhouettes.length
    : 0;
}

/**
 * Davies-Bouldin Index
 * Measures average similarity between each cluster and its most similar cluster
 * Range: 0+ (lower is better)
 */
export function calculateDaviesBouldinIndex(
  pointMap: Map<string, DataPoint>,
  clusters: Cluster[],
  _assignments: Map<string, number>
): number {
  if (clusters.length < 2) return 0;

  // Calculate scatter (average distance from centroid) for each cluster
  const scatters: number[] = [];
  for (const cluster of clusters) {
    let sumDist = 0;
    for (const memberId of cluster.members) {
      const point = pointMap.get(memberId);
      if (point) {
        sumDist += euclideanDistance(point.embedding, cluster.centroid);
      }
    }
    scatters.push(cluster.size > 0 ? sumDist / cluster.size : 0);
  }

  // Calculate Davies-Bouldin index
  let dbSum = 0;

  for (let i = 0; i < clusters.length; i++) {
    let maxRatio = 0;

    for (let j = 0; j < clusters.length; j++) {
      if (i === j) continue;

      const centroidDist = euclideanDistance(
        clusters[i].centroid,
        clusters[j].centroid
      );

      if (centroidDist > 0) {
        const ratio = (scatters[i] + scatters[j]) / centroidDist;
        maxRatio = Math.max(maxRatio, ratio);
      }
    }

    dbSum += maxRatio;
  }

  return dbSum / clusters.length;
}

/**
 * Calinski-Harabasz Index (Variance Ratio Criterion)
 * Ratio of between-cluster dispersion to within-cluster dispersion
 * Range: 0+ (higher is better)
 */
export function calculateCalinskiHarabaszIndex(
  pointMap: Map<string, DataPoint>,
  clusters: Cluster[],
  assignments: Map<string, number>
): number {
  const k = clusters.length;
  if (k < 2) return 0;

  const n = assignments.size;
  if (n <= k) return 0;

  // Calculate global centroid
  const dim = clusters[0]?.centroid?.length || 0;
  const globalCentroid = new Array(dim).fill(0);
  let totalPoints = 0;

  for (const point of pointMap.values()) {
    for (let d = 0; d < dim; d++) {
      globalCentroid[d] += point.embedding[d] || 0;
    }
    totalPoints++;
  }

  for (let d = 0; d < dim; d++) {
    globalCentroid[d] /= totalPoints;
  }

  // Calculate between-cluster sum of squares (BGSS)
  let bgss = 0;
  for (const cluster of clusters) {
    const dist = squaredEuclideanDistance(cluster.centroid, globalCentroid);
    bgss += cluster.size * dist;
  }

  // Calculate within-cluster sum of squares (WGSS)
  let wgss = 0;
  for (const cluster of clusters) {
    for (const memberId of cluster.members) {
      const point = pointMap.get(memberId);
      if (point) {
        wgss += squaredEuclideanDistance(point.embedding, cluster.centroid);
      }
    }
  }

  // Calinski-Harabasz index
  if (wgss === 0) return 0;
  return (bgss / (k - 1)) / (wgss / (n - k));
}

/**
 * Dunn Index
 * Ratio of minimum inter-cluster distance to maximum intra-cluster diameter
 * Range: 0+ (higher is better)
 */
export function calculateDunnIndex(
  pointMap: Map<string, DataPoint>,
  clusters: Cluster[],
  _assignments: Map<string, number>
): number {
  if (clusters.length < 2) return 0;

  // Calculate minimum inter-cluster distance
  let minInterCluster = Infinity;

  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      // Find minimum distance between any two points in different clusters
      for (const idA of clusters[i].members) {
        const pointA = pointMap.get(idA);
        if (!pointA) continue;

        for (const idB of clusters[j].members) {
          const pointB = pointMap.get(idB);
          if (!pointB) continue;

          const dist = euclideanDistance(pointA.embedding, pointB.embedding);
          minInterCluster = Math.min(minInterCluster, dist);
        }
      }
    }
  }

  // Calculate maximum intra-cluster diameter
  let maxIntraCluster = 0;

  for (const cluster of clusters) {
    for (let i = 0; i < cluster.members.length; i++) {
      const pointA = pointMap.get(cluster.members[i]);
      if (!pointA) continue;

      for (let j = i + 1; j < cluster.members.length; j++) {
        const pointB = pointMap.get(cluster.members[j]);
        if (!pointB) continue;

        const dist = euclideanDistance(pointA.embedding, pointB.embedding);
        maxIntraCluster = Math.max(maxIntraCluster, dist);
      }
    }
  }

  // Dunn index
  if (maxIntraCluster === 0) return 0;
  return minInterCluster / maxIntraCluster;
}

/**
 * Calculate cluster density
 * Points per unit volume (approximated)
 */
export function calculateClusterDensity(
  pointMap: Map<string, DataPoint>,
  cluster: Cluster
): number {
  if (cluster.size === 0) return 0;

  // Calculate average distance from centroid
  let sumDist = 0;
  for (const memberId of cluster.members) {
    const point = pointMap.get(memberId);
    if (point) {
      sumDist += euclideanDistance(point.embedding, cluster.centroid);
    }
  }

  const avgRadius = sumDist / cluster.size;
  if (avgRadius === 0) return cluster.size;

  // Density = points / approximate volume
  const dim = cluster.centroid.length;
  const volume = Math.pow(avgRadius, dim);
  return cluster.size / volume;
}

/**
 * Validate cluster quality
 * Returns true if cluster passes quality thresholds
 */
export function isValidCluster(
  validity: ClusterValidityIndices,
  thresholds = {
    minSilhouette: 0.2,
    maxDaviesBouldin: 2.0,
    minCalinskiHarabasz: 10,
    minDunn: 0.1,
  }
): boolean {
  return (
    validity.silhouette >= thresholds.minSilhouette &&
    validity.daviesBouldin <= thresholds.maxDaviesBouldin &&
    validity.calinskiHarabasz >= thresholds.minCalinskiHarabasz &&
    validity.dunn >= thresholds.minDunn
  );
}

/**
 * Get cluster quality rating based on validity indices
 */
export function getClusterQualityRating(
  validity: ClusterValidityIndices
): 'excellent' | 'good' | 'fair' | 'poor' {
  const silhouetteScore =
    validity.silhouette > 0.7
      ? 4
      : validity.silhouette > 0.5
        ? 3
        : validity.silhouette > 0.25
          ? 2
          : 1;

  const dbScore =
    validity.daviesBouldin < 0.5
      ? 4
      : validity.daviesBouldin < 1.0
        ? 3
        : validity.daviesBouldin < 2.0
          ? 2
          : 1;

  const avgScore = (silhouetteScore + dbScore) / 2;

  if (avgScore >= 3.5) return 'excellent';
  if (avgScore >= 2.5) return 'good';
  if (avgScore >= 1.5) return 'fair';
  return 'poor';
}

// ============================================================================
// Utility Functions
// ============================================================================

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - (b[i] || 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function squaredEuclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - (b[i] || 0);
    sum += diff * diff;
  }
  return sum;
}
