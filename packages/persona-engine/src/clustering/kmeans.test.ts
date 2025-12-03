/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { KMeans, buildClusters, findOptimalK } from './kmeans.js';
import type { DataPoint } from '../types.js';

describe('KMeans', () => {
  function generateClusteredData(
    centers: number[][],
    pointsPerCluster: number,
    noise: number
  ): number[][] {
    const data: number[][] = [];
    for (const center of centers) {
      for (let i = 0; i < pointsPerCluster; i++) {
        const point = center.map(
          (c) => c + (Math.random() - 0.5) * noise * 2
        );
        data.push(point);
      }
    }
    return data;
  }

  describe('fit', () => {
    it('should separate distinct clusters', () => {
      const data = generateClusteredData(
        [
          [0, 0],
          [10, 10],
          [20, 0],
        ],
        20,
        1
      );

      const kmeans = new KMeans(3);
      const assignments = kmeans.fit(data);

      // All points should be assigned
      expect(assignments.length).toBe(60);

      // Should have 3 distinct clusters
      const uniqueClusters = new Set(assignments);
      expect(uniqueClusters.size).toBe(3);

      // Points from the same original cluster should be in the same k-means cluster
      const cluster0 = new Set(assignments.slice(0, 20));
      const cluster1 = new Set(assignments.slice(20, 40));
      const cluster2 = new Set(assignments.slice(40, 60));

      // Each original cluster should map to a single k-means cluster
      expect(cluster0.size).toBe(1);
      expect(cluster1.size).toBe(1);
      expect(cluster2.size).toBe(1);
    });

    it('should handle single point', () => {
      const kmeans = new KMeans(1);
      const assignments = kmeans.fit([[1, 2]]);
      expect(assignments).toEqual([0]);
    });

    it('should handle fewer points than k', () => {
      const kmeans = new KMeans(5);
      const assignments = kmeans.fit([
        [1, 1],
        [2, 2],
      ]);
      expect(assignments.length).toBe(2);
    });
  });

  describe('getCentroids', () => {
    it('should return k centroids', () => {
      const data = generateClusteredData(
        [
          [0, 0],
          [10, 10],
        ],
        10,
        1
      );

      const kmeans = new KMeans(2);
      kmeans.fit(data);
      const centroids = kmeans.getCentroids();

      expect(centroids.length).toBe(2);
      expect(centroids[0].length).toBe(2); // 2D data
    });

    it('should have centroids near cluster centers', () => {
      const data = generateClusteredData([[0, 0], [100, 100]], 50, 2);

      const kmeans = new KMeans(2);
      kmeans.fit(data);
      const centroids = kmeans.getCentroids();

      // One centroid should be near [0, 0], the other near [100, 100]
      const nearOrigin = centroids.some(
        (c) => Math.abs(c[0]) < 10 && Math.abs(c[1]) < 10
      );
      const nearFar = centroids.some(
        (c) => Math.abs(c[0] - 100) < 10 && Math.abs(c[1] - 100) < 10
      );

      expect(nearOrigin).toBe(true);
      expect(nearFar).toBe(true);
    });
  });

  describe('calculateInertia', () => {
    it('should decrease with more clusters', () => {
      const data = generateClusteredData(
        [
          [0, 0],
          [10, 10],
          [20, 20],
        ],
        20,
        1
      );

      const kmeans2 = new KMeans(2);
      const assignments2 = kmeans2.fit(data);
      const inertia2 = kmeans2.calculateInertia(data, assignments2);

      const kmeans3 = new KMeans(3);
      const assignments3 = kmeans3.fit(data);
      const inertia3 = kmeans3.calculateInertia(data, assignments3);

      expect(inertia3).toBeLessThan(inertia2);
    });
  });
});

describe('buildClusters', () => {
  it('should create cluster objects from assignments', () => {
    const dataPoints: DataPoint[] = [
      { id: 'a', embedding: [0, 0], metadata: {} },
      { id: 'b', embedding: [1, 1], metadata: {} },
      { id: 'c', embedding: [10, 10], metadata: {} },
      { id: 'd', embedding: [11, 11], metadata: {} },
    ];

    const assignments = [0, 0, 1, 1];
    const centroids = [
      [0.5, 0.5],
      [10.5, 10.5],
    ];

    const clusters = buildClusters(dataPoints, assignments, centroids);

    expect(clusters.length).toBe(2);
    expect(clusters[0].members).toContain('a');
    expect(clusters[0].members).toContain('b');
    expect(clusters[1].members).toContain('c');
    expect(clusters[1].members).toContain('d');
  });
});

describe('findOptimalK', () => {
  it('should find reasonable k for distinct clusters', () => {
    const data = [
      // Cluster 1
      ...Array.from({ length: 20 }, () => [
        Math.random() * 2,
        Math.random() * 2,
      ]),
      // Cluster 2
      ...Array.from({ length: 20 }, () => [
        10 + Math.random() * 2,
        10 + Math.random() * 2,
      ]),
      // Cluster 3
      ...Array.from({ length: 20 }, () => [
        20 + Math.random() * 2,
        Math.random() * 2,
      ]),
    ];

    const optimalK = findOptimalK(data, 2, 8);

    // Should find k close to 3
    expect(optimalK).toBeGreaterThanOrEqual(2);
    expect(optimalK).toBeLessThanOrEqual(5);
  });
});
