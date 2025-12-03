/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Monte Carlo Persona Validation
 * Perturb embeddings and re-cluster to measure persona robustness
 */

import type {
  Cluster,
  DataPoint,
  MonteCarloResult,
  ClusteringAlgorithm,
} from '../types.js';
import { KMeans, buildClusters } from '../clustering/kmeans.js';
import { HDBSCAN, buildHDBSCANClusters } from '../clustering/hdbscan.js';
import { normalRandom } from '../statistics/distributions.js';

/**
 * Configuration for Monte Carlo validation
 */
export interface MonteCarloConfig {
  iterations: number; // Number of perturbation runs
  perturbationMagnitude: number; // Standard deviation of perturbation noise
  algorithm: ClusteringAlgorithm;
  k?: number; // For k-means
  minClusterSize?: number; // For HDBSCAN
  convergenceThreshold: number; // Stop early if results stabilize
}

const DEFAULT_MONTE_CARLO_CONFIG: MonteCarloConfig = {
  iterations: 100,
  perturbationMagnitude: 0.1,
  algorithm: 'kmeans',
  convergenceThreshold: 0.95,
  minClusterSize: 5,
};

/**
 * Monte Carlo validator for persona robustness
 */
export class MonteCarloValidator {
  private config: MonteCarloConfig;
  private membershipMatrix: number[][] = [];
  private clusterPersistence: number[] = [];

  constructor(config: Partial<MonteCarloConfig> = {}) {
    this.config = { ...DEFAULT_MONTE_CARLO_CONFIG, ...config };
  }

  /**
   * Run Monte Carlo validation on clustered data
   */
  validate(
    dataPoints: DataPoint[],
    originalClusters: Cluster[]
  ): MonteCarloResult {
    const n = dataPoints.length;
    const k = originalClusters.length;

    if (n === 0 || k === 0) {
      return {
        robustnessScore: 0,
        membershipMatrix: [],
        clusterPersistence: [],
        convergenceIterations: 0,
      };
    }

    // Initialize membership matrix: n x k
    // Tracks how often each point lands in each cluster
    this.membershipMatrix = Array.from({ length: n }, () =>
      new Array(k).fill(0)
    );

    // Track cluster persistence
    this.clusterPersistence = new Array(k).fill(0);

    // Build original assignment map
    const originalAssignments = this.buildAssignmentMap(
      dataPoints,
      originalClusters
    );

    // Run Monte Carlo iterations
    let convergenceReached = false;
    let lastRobustness = 0;
    let iteration = 0;

    for (iteration = 0; iteration < this.config.iterations; iteration++) {
      // Perturb embeddings
      const perturbedPoints = this.perturbEmbeddings(dataPoints);

      // Re-cluster
      const newClusters = this.cluster(perturbedPoints, k);

      // Match new clusters to original clusters
      const clusterMapping = this.matchClusters(originalClusters, newClusters);

      // Update membership matrix
      this.updateMembershipMatrix(
        perturbedPoints,
        newClusters,
        clusterMapping,
        originalAssignments
      );

      // Update cluster persistence
      this.updateClusterPersistence(newClusters, clusterMapping);

      // Check for convergence
      const currentRobustness = this.calculateRobustness(
        originalAssignments,
        iteration + 1
      );
      if (
        iteration > 10 &&
        Math.abs(currentRobustness - lastRobustness) < 0.001
      ) {
        if (currentRobustness > this.config.convergenceThreshold) {
          convergenceReached = true;
          break;
        }
      }
      lastRobustness = currentRobustness;
    }

    // Normalize membership matrix
    const iterations = iteration + 1;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < k; j++) {
        this.membershipMatrix[i][j] /= iterations;
      }
    }

    // Normalize cluster persistence
    for (let j = 0; j < k; j++) {
      this.clusterPersistence[j] /= iterations;
    }

    return {
      robustnessScore: this.calculateRobustness(originalAssignments, iterations),
      membershipMatrix: this.membershipMatrix,
      clusterPersistence: this.clusterPersistence,
      convergenceIterations: convergenceReached ? iteration + 1 : this.config.iterations,
    };
  }

  /**
   * Add Gaussian noise to embeddings
   */
  private perturbEmbeddings(dataPoints: DataPoint[]): DataPoint[] {
    return dataPoints.map((point) => ({
      ...point,
      embedding: point.embedding.map(
        (v) => v + normalRandom(0, this.config.perturbationMagnitude)
      ),
    }));
  }

  /**
   * Cluster perturbed data
   */
  private cluster(dataPoints: DataPoint[], k: number): Cluster[] {
    const embeddings = dataPoints.map((d) => d.embedding);

    if (this.config.algorithm === 'hdbscan') {
      const hdbscan = new HDBSCAN(this.config.minClusterSize);
      const assignments = hdbscan.fit(embeddings);
      return buildHDBSCANClusters(dataPoints, assignments);
    }

    const kmeans = new KMeans(k);
    const assignments = kmeans.fit(embeddings);
    return buildClusters(dataPoints, assignments, kmeans.getCentroids());
  }

  /**
   * Build assignment map from data points to cluster indices
   */
  private buildAssignmentMap(
    dataPoints: DataPoint[],
    clusters: Cluster[]
  ): Map<string, number> {
    const assignments = new Map<string, number>();
    for (let i = 0; i < clusters.length; i++) {
      for (const memberId of clusters[i].members) {
        assignments.set(memberId, i);
      }
    }
    return assignments;
  }

  /**
   * Match new clusters to original clusters using centroid similarity
   */
  private matchClusters(
    originalClusters: Cluster[],
    newClusters: Cluster[]
  ): Map<number, number> {
    const mapping = new Map<number, number>();
    const used = new Set<number>();

    // Greedy matching based on centroid distance
    for (let i = 0; i < newClusters.length; i++) {
      let bestMatch = -1;
      let bestDistance = Infinity;

      for (let j = 0; j < originalClusters.length; j++) {
        if (used.has(j)) continue;

        const dist = this.euclideanDistance(
          newClusters[i].centroid,
          originalClusters[j].centroid
        );

        if (dist < bestDistance) {
          bestDistance = dist;
          bestMatch = j;
        }
      }

      if (bestMatch >= 0) {
        mapping.set(i, bestMatch);
        used.add(bestMatch);
      }
    }

    return mapping;
  }

  /**
   * Update membership matrix based on new clustering
   */
  private updateMembershipMatrix(
    dataPoints: DataPoint[],
    newClusters: Cluster[],
    clusterMapping: Map<number, number>,
    _originalAssignments: Map<string, number>
  ): void {
    // Build index map from data point id to index
    const indexMap = new Map<string, number>();
    for (let i = 0; i < dataPoints.length; i++) {
      indexMap.set(dataPoints[i].id, i);
    }

    // Update membership matrix
    for (let newClusterIdx = 0; newClusterIdx < newClusters.length; newClusterIdx++) {
      const originalClusterIdx = clusterMapping.get(newClusterIdx);
      if (originalClusterIdx === undefined) continue;

      for (const memberId of newClusters[newClusterIdx].members) {
        const pointIdx = indexMap.get(memberId);
        if (pointIdx !== undefined) {
          this.membershipMatrix[pointIdx][originalClusterIdx]++;
        }
      }
    }
  }

  /**
   * Update cluster persistence counts
   */
  private updateClusterPersistence(
    newClusters: Cluster[],
    clusterMapping: Map<number, number>
  ): void {
    for (const originalIdx of clusterMapping.values()) {
      this.clusterPersistence[originalIdx]++;
    }
  }

  /**
   * Calculate robustness score
   * Average probability of points staying in their original cluster
   */
  private calculateRobustness(
    originalAssignments: Map<string, number>,
    iterations: number
  ): number {
    let sumProbability = 0;
    let count = 0;

    const pointIds = Array.from(originalAssignments.keys());
    for (let i = 0; i < pointIds.length; i++) {
      const originalCluster = originalAssignments.get(pointIds[i]);
      if (originalCluster !== undefined) {
        sumProbability += this.membershipMatrix[i][originalCluster] / iterations;
        count++;
      }
    }

    return count > 0 ? sumProbability / count : 0;
  }

  /**
   * Get detailed robustness scores per cluster
   */
  getClusterRobustness(
    originalClusters: Cluster[],
    dataPoints: DataPoint[]
  ): Map<string, number> {
    const robustness = new Map<string, number>();
    const indexMap = new Map<string, number>();

    for (let i = 0; i < dataPoints.length; i++) {
      indexMap.set(dataPoints[i].id, i);
    }

    for (let j = 0; j < originalClusters.length; j++) {
      const cluster = originalClusters[j];
      let sumProbability = 0;

      for (const memberId of cluster.members) {
        const idx = indexMap.get(memberId);
        if (idx !== undefined) {
          sumProbability += this.membershipMatrix[idx][j];
        }
      }

      robustness.set(
        cluster.id,
        cluster.size > 0 ? sumProbability / cluster.size : 0
      );
    }

    return robustness;
  }

  /**
   * Get cluster persistence scores
   */
  getClusterPersistence(): number[] {
    return this.clusterPersistence;
  }

  /**
   * Get membership probability matrix
   */
  getMembershipMatrix(): number[][] {
    return this.membershipMatrix;
  }

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
 * Quick utility to run Monte Carlo validation
 */
export function runMonteCarloValidation(
  dataPoints: DataPoint[],
  clusters: Cluster[],
  config?: Partial<MonteCarloConfig>
): MonteCarloResult {
  const validator = new MonteCarloValidator(config);
  return validator.validate(dataPoints, clusters);
}

/**
 * Interpret robustness score
 */
export function interpretRobustnessScore(score: number): {
  rating: 'strong' | 'moderate' | 'weak' | 'unstable';
  description: string;
} {
  if (score >= 0.9) {
    return {
      rating: 'strong',
      description: 'Highly robust persona with consistent membership across perturbations',
    };
  }
  if (score >= 0.7) {
    return {
      rating: 'moderate',
      description: 'Reasonably robust persona that maintains structure under noise',
    };
  }
  if (score >= 0.5) {
    return {
      rating: 'weak',
      description: 'Moderately stable persona that may shift under different conditions',
    };
  }
  return {
    rating: 'unstable',
    description: 'Unreliable persona that changes significantly with data perturbations',
  };
}
