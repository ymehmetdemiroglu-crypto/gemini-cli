/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import {
  calculateClusterValidityIndices,
  calculateSilhouetteScore,
  calculateDaviesBouldinIndex,
  isValidCluster,
  getClusterQualityRating,
} from './cluster-validity.js';
import type { DataPoint, Cluster } from '../types.js';

describe('cluster-validity', () => {
  function createTestData(): { dataPoints: DataPoint[]; clusters: Cluster[] } {
    // Create well-separated clusters
    const dataPoints: DataPoint[] = [];

    // Cluster 1: centered at [0, 0]
    for (let i = 0; i < 20; i++) {
      dataPoints.push({
        id: `c1_${i}`,
        embedding: [Math.random() - 0.5, Math.random() - 0.5],
        metadata: {},
      });
    }

    // Cluster 2: centered at [10, 10]
    for (let i = 0; i < 20; i++) {
      dataPoints.push({
        id: `c2_${i}`,
        embedding: [10 + Math.random() - 0.5, 10 + Math.random() - 0.5],
        metadata: {},
      });
    }

    const clusters: Cluster[] = [
      {
        id: 'cluster_0',
        centroid: [0, 0],
        members: dataPoints.slice(0, 20).map((p) => p.id),
        size: 20,
      },
      {
        id: 'cluster_1',
        centroid: [10, 10],
        members: dataPoints.slice(20).map((p) => p.id),
        size: 20,
      },
    ];

    return { dataPoints, clusters };
  }

  describe('calculateClusterValidityIndices', () => {
    it('should calculate all indices', () => {
      const { dataPoints, clusters } = createTestData();
      const indices = calculateClusterValidityIndices(dataPoints, clusters);

      expect(indices.silhouette).toBeDefined();
      expect(indices.daviesBouldin).toBeDefined();
      expect(indices.calinskiHarabasz).toBeDefined();
      expect(indices.dunn).toBeDefined();
    });

    it('should return high silhouette for well-separated clusters', () => {
      const { dataPoints, clusters } = createTestData();
      const indices = calculateClusterValidityIndices(dataPoints, clusters);

      // Well-separated clusters should have silhouette close to 1
      expect(indices.silhouette).toBeGreaterThan(0.5);
    });

    it('should return low Davies-Bouldin for well-separated clusters', () => {
      const { dataPoints, clusters } = createTestData();
      const indices = calculateClusterValidityIndices(dataPoints, clusters);

      // Well-separated clusters should have low DB index
      expect(indices.daviesBouldin).toBeLessThan(1);
    });
  });

  describe('calculateSilhouetteScore', () => {
    it('should return value between -1 and 1', () => {
      const { dataPoints, clusters } = createTestData();
      const pointMap = new Map(dataPoints.map((p) => [p.id, p]));
      const assignments = new Map<string, number>();
      clusters.forEach((c, i) => c.members.forEach((m) => assignments.set(m, i)));

      const score = calculateSilhouetteScore(pointMap, clusters, assignments);

      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should return 0 for single cluster', () => {
      const dataPoints: DataPoint[] = Array.from({ length: 10 }, (_, i) => ({
        id: `p${i}`,
        embedding: [Math.random(), Math.random()],
        metadata: {},
      }));

      const clusters: Cluster[] = [
        {
          id: 'cluster_0',
          centroid: [0.5, 0.5],
          members: dataPoints.map((p) => p.id),
          size: 10,
        },
      ];

      const pointMap = new Map(dataPoints.map((p) => [p.id, p]));
      const assignments = new Map<string, number>();
      clusters[0].members.forEach((m) => assignments.set(m, 0));

      const score = calculateSilhouetteScore(pointMap, clusters, assignments);

      expect(score).toBe(0);
    });
  });

  describe('calculateDaviesBouldinIndex', () => {
    it('should return positive value', () => {
      const { dataPoints, clusters } = createTestData();
      const pointMap = new Map(dataPoints.map((p) => [p.id, p]));
      const assignments = new Map<string, number>();
      clusters.forEach((c, i) => c.members.forEach((m) => assignments.set(m, i)));

      const index = calculateDaviesBouldinIndex(pointMap, clusters, assignments);

      expect(index).toBeGreaterThanOrEqual(0);
    });
  });

  describe('isValidCluster', () => {
    it('should validate good clusters', () => {
      const goodIndices = {
        silhouette: 0.7,
        daviesBouldin: 0.5,
        calinskiHarabasz: 100,
        dunn: 0.5,
      };

      expect(isValidCluster(goodIndices)).toBe(true);
    });

    it('should reject poor clusters', () => {
      const poorIndices = {
        silhouette: -0.1,
        daviesBouldin: 5,
        calinskiHarabasz: 5,
        dunn: 0.01,
      };

      expect(isValidCluster(poorIndices)).toBe(false);
    });
  });

  describe('getClusterQualityRating', () => {
    it('should rate excellent clusters', () => {
      const excellent = {
        silhouette: 0.85,
        daviesBouldin: 0.3,
        calinskiHarabasz: 500,
        dunn: 0.8,
      };

      expect(getClusterQualityRating(excellent)).toBe('excellent');
    });

    it('should rate poor clusters', () => {
      const poor = {
        silhouette: 0.1,
        daviesBouldin: 3,
        calinskiHarabasz: 10,
        dunn: 0.05,
      };

      expect(getClusterQualityRating(poor)).toBe('poor');
    });
  });
});
