/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import {
  normalRandom,
  normalPDF,
  normalCDF,
  normalQuantile,
  gamma,
  chiSquareCDF,
  tCDF,
  createDistributionFromData,
  createCategoricalDistribution,
  createHistogramDistribution,
  createBetaDistribution,
  distributionMean,
  distributionVariance,
} from './distributions.js';

describe('distributions', () => {
  describe('normalRandom', () => {
    it('should generate values with correct mean', () => {
      const samples: number[] = [];
      for (let i = 0; i < 10000; i++) {
        samples.push(normalRandom(5, 1));
      }
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      expect(mean).toBeCloseTo(5, 1);
    });

    it('should generate values with correct std', () => {
      const samples: number[] = [];
      for (let i = 0; i < 10000; i++) {
        samples.push(normalRandom(0, 2));
      }
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const variance =
        samples.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) /
        (samples.length - 1);
      expect(Math.sqrt(variance)).toBeCloseTo(2, 0);
    });
  });

  describe('normalPDF', () => {
    it('should return correct value at mean', () => {
      const pdf = normalPDF(0, 0, 1);
      expect(pdf).toBeCloseTo(0.3989, 3);
    });

    it('should be symmetric', () => {
      expect(normalPDF(-1, 0, 1)).toBeCloseTo(normalPDF(1, 0, 1), 10);
    });
  });

  describe('normalCDF', () => {
    it('should return 0.5 at mean', () => {
      expect(normalCDF(0, 0, 1)).toBeCloseTo(0.5, 5);
    });

    it('should return ~0.8413 at z=1', () => {
      expect(normalCDF(1, 0, 1)).toBeCloseTo(0.8413, 2);
    });

    it('should return ~0.9772 at z=2', () => {
      expect(normalCDF(2, 0, 1)).toBeCloseTo(0.9772, 2);
    });
  });

  describe('normalQuantile', () => {
    it('should return 0 at p=0.5', () => {
      expect(normalQuantile(0.5)).toBeCloseTo(0, 2);
    });

    it('should return ~1.645 at p=0.95', () => {
      expect(normalQuantile(0.95)).toBeCloseTo(1.645, 1);
    });
  });

  describe('gamma', () => {
    it('should return factorial for integers', () => {
      expect(gamma(5)).toBeCloseTo(24, 1); // (5-1)! = 24
      expect(gamma(6)).toBeCloseTo(120, 1); // (6-1)! = 120
    });
  });

  describe('chiSquareCDF', () => {
    it('should return ~0.95 for chi-square with df=1 at x≈3.84', () => {
      expect(chiSquareCDF(3.84, 1)).toBeCloseTo(0.95, 1);
    });
  });

  describe('tCDF', () => {
    it('should return 0.5 at t=0', () => {
      expect(tCDF(0, 10)).toBeCloseTo(0.5, 5);
    });
  });

  describe('createDistributionFromData', () => {
    it('should calculate correct mean and std', () => {
      const data = [1, 2, 3, 4, 5];
      const dist = createDistributionFromData(data);
      expect(dist.mean).toBeCloseTo(3, 10);
      expect(dist.std).toBeCloseTo(1.58, 1);
    });
  });

  describe('createCategoricalDistribution', () => {
    it('should create correct frequency distribution', () => {
      const data = ['a', 'a', 'b', 'b', 'b', 'c'];
      const dist = createCategoricalDistribution(data);
      expect(dist.categories?.get('a')).toBeCloseTo(2 / 6, 10);
      expect(dist.categories?.get('b')).toBeCloseTo(3 / 6, 10);
      expect(dist.categories?.get('c')).toBeCloseTo(1 / 6, 10);
    });
  });

  describe('createHistogramDistribution', () => {
    it('should create histogram with correct bins', () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const dist = createHistogramDistribution(data, 5);
      expect(dist.bins?.length).toBe(6); // 5 bins = 6 edges
      expect(dist.frequencies?.length).toBe(5);
    });
  });

  describe('createBetaDistribution', () => {
    it('should create Beta distribution with correct parameters', () => {
      const dist = createBetaDistribution(8, 2, 1, 1);
      expect(dist.alpha).toBe(9);
      expect(dist.beta).toBe(3);
    });
  });

  describe('distributionMean', () => {
    it('should calculate normal distribution mean', () => {
      const dist = createDistributionFromData([1, 2, 3, 4, 5]);
      expect(distributionMean(dist)).toBeCloseTo(3, 10);
    });

    it('should calculate beta distribution mean', () => {
      const dist = createBetaDistribution(4, 1, 1, 1);
      // Mean = alpha / (alpha + beta) = 5 / 7
      expect(distributionMean(dist)).toBeCloseTo(5 / 7, 2);
    });
  });

  describe('distributionVariance', () => {
    it('should calculate normal distribution variance', () => {
      const dist = { type: 'normal' as const, mean: 0, std: 2 };
      expect(distributionVariance(dist)).toBeCloseTo(4, 10);
    });
  });
});
