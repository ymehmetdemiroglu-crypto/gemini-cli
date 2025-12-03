/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Bootstrapped sampling for cluster stability analysis
 */

import type {
  BootstrapResult,
  Cluster,
  ClusterStability,
  DataPoint,
  ClusteringAlgorithm,
} from '../types.js';
import { KMeans, buildClusters, findOptimalK } from './kmeans.js';
import { HDBSCAN, buildHDBSCANClusters } from './hdbscan.js';

/**
 * Configuration for bootstrap sampling
 */
export interface BootstrapConfig {
  numSamples: number; // Number of bootstrap iterations
  sampleSize: number; // Size of each bootstrap sample (as fraction or absolute)
  sampleAsFraction: boolean; // If true, sampleSize is treated as fraction of data
  algorithm: ClusteringAlgorithm;
  kmeansK?: number; // If not provided, will be determined automatically
  minClusterSize?: number; // For HDBSCAN
  seed?: number; // Random seed for reproducibility
}

const DEFAULT_BOOTSTRAP_CONFIG: BootstrapConfig = {
  numSamples: 100,
  sampleSize: 0.8,
  sampleAsFraction: true,
  algorithm: 'kmeans',
  minClusterSize: 5,
};

/**
 * Bootstrapped clustering for stability analysis
 */
export class BootstrappedClustering {
  private config: BootstrapConfig;
  private results: BootstrapResult[] = [];
  private dataPoints: DataPoint[] = [];

  constructor(config: Partial<BootstrapConfig> = {}) {
    this.config = { ...DEFAULT_BOOTSTRAP_CONFIG, ...config };
  }

  /**
   * Run bootstrapped clustering
   */
  run(dataPoints: DataPoint[]): BootstrapResult[] {
    this.dataPoints = dataPoints;
    this.results = [];

    const n = dataPoints.length;
    const sampleSize = this.config.sampleAsFraction
      ? Math.floor(n * this.config.sampleSize)
      : Math.min(this.config.sampleSize, n);

    // Determine k for k-means if not provided
    let k = this.config.kmeansK;
    if (!k && this.config.algorithm === 'kmeans') {
      const embeddings = dataPoints.map((d) => d.embedding);
      k = findOptimalK(embeddings, 2, Math.min(10, Math.floor(n / 5)));
    }

    // Run bootstrap iterations
    for (let i = 0; i < this.config.numSamples; i++) {
      const sampleIndices = this.sampleWithReplacement(n, sampleSize);
      const sample = sampleIndices.map((idx) => dataPoints[idx]);

      const clusters = this.clusterSample(sample, k);
      const validityIndices = this.calculateValidityIndices(
        sample.map((d) => d.embedding),
        clusters
      );

      this.results.push({
        clusters,
        validityIndices,
        sampleIndices,
      });
    }

    return this.results;
  }

  /**
   * Sample indices with replacement (bootstrap sample)
   */
  private sampleWithReplacement(n: number, size: number): number[] {
    const indices: number[] = [];
    for (let i = 0; i < size; i++) {
      indices.push(Math.floor(Math.random() * n));
    }
    return indices;
  }

  /**
   * Cluster a sample of data points
   */
  private clusterSample(sample: DataPoint[], k?: number): Cluster[] {
    const embeddings = sample.map((d) => d.embedding);

    if (this.config.algorithm === 'hdbscan') {
      const hdbscan = new HDBSCAN(this.config.minClusterSize);
      const assignments = hdbscan.fit(embeddings);
      return buildHDBSCANClusters(sample, assignments);
    }

    // K-means
    const kmeans = new KMeans(k || 3);
    const assignments = kmeans.fit(embeddings);
    return buildClusters(sample, assignments, kmeans.getCentroids());
  }

  /**
   * Calculate basic validity indices (detailed implementation in validation module)
   */
  private calculateValidityIndices(
    _embeddings: number[][],
    _clusters: Cluster[]
  ): {
    silhouette: number;
    daviesBouldin: number;
    calinskiHarabasz: number;
    dunn: number;
  } {
    // Placeholder - detailed implementation in validation module
    return {
      silhouette: 0,
      daviesBouldin: 0,
      calinskiHarabasz: 0,
      dunn: 0,
    };
  }

  /**
   * Calculate cluster stability scores across bootstrap samples
   */
  calculateStability(): Map<string, ClusterStability> {
    if (this.results.length === 0) {
      return new Map();
    }

    // Track cluster appearances and membership consistency
    const clusterAppearances = new Map<string, number>();
    const membershipCounts = new Map<string, Map<string, number>>();

    // Create cluster signatures based on top members
    for (const result of this.results) {
      for (const cluster of result.clusters) {
        const signature = this.getClusterSignature(cluster);

        clusterAppearances.set(
          signature,
          (clusterAppearances.get(signature) || 0) + 1
        );

        if (!membershipCounts.has(signature)) {
          membershipCounts.set(signature, new Map());
        }

        const counts = membershipCounts.get(signature)!;
        for (const member of cluster.members) {
          counts.set(member, (counts.get(member) || 0) + 1);
        }
      }
    }

    // Calculate stability metrics for each cluster signature
    const stabilityMap = new Map<string, ClusterStability>();

    for (const [signature, appearances] of clusterAppearances) {
      const appearanceRate = appearances / this.results.length;

      // Calculate membership consistency
      const memberCounts = membershipCounts.get(signature)!;
      const memberConsistencies: number[] = [];
      for (const count of memberCounts.values()) {
        memberConsistencies.push(count / appearances);
      }

      const avgConsistency =
        memberConsistencies.length > 0
          ? memberConsistencies.reduce((a, b) => a + b, 0) /
            memberConsistencies.length
          : 0;

      // Bootstrap confidence interval
      const sortedConsistencies = [...memberConsistencies].sort((a, b) => a - b);
      const lowerIdx = Math.floor(sortedConsistencies.length * 0.025);
      const upperIdx = Math.floor(sortedConsistencies.length * 0.975);

      const confidenceInterval: [number, number] = [
        sortedConsistencies[lowerIdx] || 0,
        sortedConsistencies[upperIdx] || 1,
      ];

      const stabilityScore = appearanceRate * avgConsistency;

      stabilityMap.set(signature, {
        stabilityScore,
        bootstrapConsistency: avgConsistency,
        confidenceInterval,
        appearanceRate,
      });
    }

    return stabilityMap;
  }

  /**
   * Get stable clusters (appearing in most bootstrap samples)
   */
  getStableClusters(minStability = 0.5): Cluster[] {
    const stability = this.calculateStability();
    const stableClusters: Cluster[] = [];

    // Get the most recent result's clusters
    if (this.results.length === 0) return [];

    const lastResult = this.results[this.results.length - 1];

    for (const cluster of lastResult.clusters) {
      const signature = this.getClusterSignature(cluster);
      const clusterStability = stability.get(signature);

      if (clusterStability && clusterStability.stabilityScore >= minStability) {
        stableClusters.push(cluster);
      }
    }

    return stableClusters;
  }

  /**
   * Create a signature for a cluster based on its characteristics
   */
  private getClusterSignature(cluster: Cluster): string {
    // Use centroid (rounded) as signature
    const roundedCentroid = cluster.centroid
      .slice(0, 5)
      .map((v) => Math.round(v * 100) / 100);
    return `c_${roundedCentroid.join('_')}_s${cluster.size}`;
  }

  /**
   * Get aggregated validity indices across all bootstrap samples
   */
  getAggregatedValidityIndices(): {
    mean: {
      silhouette: number;
      daviesBouldin: number;
      calinskiHarabasz: number;
      dunn: number;
    };
    std: {
      silhouette: number;
      daviesBouldin: number;
      calinskiHarabasz: number;
      dunn: number;
    };
  } {
    if (this.results.length === 0) {
      return {
        mean: { silhouette: 0, daviesBouldin: 0, calinskiHarabasz: 0, dunn: 0 },
        std: { silhouette: 0, daviesBouldin: 0, calinskiHarabasz: 0, dunn: 0 },
      };
    }

    const silhouettes = this.results.map((r) => r.validityIndices.silhouette);
    const daviesBouldin = this.results.map((r) => r.validityIndices.daviesBouldin);
    const calinskiHarabasz = this.results.map((r) => r.validityIndices.calinskiHarabasz);
    const dunns = this.results.map((r) => r.validityIndices.dunn);

    return {
      mean: {
        silhouette: this.mean(silhouettes),
        daviesBouldin: this.mean(daviesBouldin),
        calinskiHarabasz: this.mean(calinskiHarabasz),
        dunn: this.mean(dunns),
      },
      std: {
        silhouette: this.std(silhouettes),
        daviesBouldin: this.std(daviesBouldin),
        calinskiHarabasz: this.std(calinskiHarabasz),
        dunn: this.std(dunns),
      },
    };
  }

  private mean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private std(values: number[]): number {
    if (values.length < 2) return 0;
    const m = this.mean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - m, 2), 0) / (values.length - 1);
    return Math.sqrt(variance);
  }

  /**
   * Get all bootstrap results
   */
  getResults(): BootstrapResult[] {
    return this.results;
  }
}

/**
 * Quick utility to run bootstrap clustering and get stable clusters
 */
export function runBootstrapClustering(
  dataPoints: DataPoint[],
  config?: Partial<BootstrapConfig>
): { clusters: Cluster[]; stability: Map<string, ClusterStability> } {
  const bootstrap = new BootstrappedClustering(config);
  bootstrap.run(dataPoints);

  return {
    clusters: bootstrap.getStableClusters(),
    stability: bootstrap.calculateStability(),
  };
}
