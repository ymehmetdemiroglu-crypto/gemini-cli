/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import {
  calculateStats,
  oneSampleTTest,
  twoSampleTTest,
  chiSquareTest,
  chiSquareGoodnessOfFit,
  oneWayANOVA,
  mannWhitneyU,
  kolmogorovSmirnovTest,
  confidenceIntervalMean,
  confidenceIntervalProportion,
} from './hypothesis-testing.js';

describe('hypothesis-testing', () => {
  describe('calculateStats', () => {
    it('should calculate basic statistics correctly', () => {
      const data = [2, 4, 6, 8, 10];
      const stats = calculateStats(data);

      expect(stats.n).toBe(5);
      expect(stats.mean).toBe(6);
      expect(stats.min).toBe(2);
      expect(stats.max).toBe(10);
      expect(stats.variance).toBeCloseTo(10, 10);
    });

    it('should handle empty array', () => {
      const stats = calculateStats([]);
      expect(stats.n).toBe(0);
      expect(stats.mean).toBe(0);
    });
  });

  describe('oneSampleTTest', () => {
    it('should detect significant difference from hypothesized mean', () => {
      // Data with mean ≈ 10
      const data = [9, 10, 11, 10, 9, 11, 10, 10, 9, 11];
      const result = oneSampleTTest(data, 5);

      expect(result.isSignificant).toBe(true);
      expect(result.pValue).toBeLessThan(0.05);
    });

    it('should not detect difference when data matches hypothesis', () => {
      const data = [9, 10, 11, 10, 9, 11, 10, 10, 9, 11];
      const result = oneSampleTTest(data, 10);

      expect(result.isSignificant).toBe(false);
      expect(result.pValue).toBeGreaterThan(0.05);
    });
  });

  describe('twoSampleTTest', () => {
    it('should detect significant difference between groups', () => {
      const group1 = [1, 2, 3, 2, 1, 2, 3, 2, 1, 2];
      const group2 = [8, 9, 10, 9, 8, 9, 10, 9, 8, 9];
      const result = twoSampleTTest(group1, group2);

      expect(result.isSignificant).toBe(true);
      expect(result.pValue).toBeLessThan(0.01);
    });

    it('should not detect difference for similar groups', () => {
      const group1 = [5, 6, 5, 6, 5, 6, 5, 6, 5, 6];
      const group2 = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5];
      const result = twoSampleTTest(group1, group2);

      expect(result.pValue).toBeGreaterThan(0.1);
    });
  });

  describe('chiSquareTest', () => {
    it('should detect significant association', () => {
      // Strong association between rows and columns
      const observed = [
        [50, 10],
        [10, 50],
      ];
      const result = chiSquareTest(observed);

      expect(result.isSignificant).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it('should not detect association for uniform distribution', () => {
      const observed = [
        [25, 25],
        [25, 25],
      ];
      const result = chiSquareTest(observed);

      expect(result.isSignificant).toBe(false);
    });
  });

  describe('chiSquareGoodnessOfFit', () => {
    it('should detect deviation from expected distribution', () => {
      const observed = [100, 50, 25, 25]; // Not uniform
      const expected = [50, 50, 50, 50]; // Uniform
      const result = chiSquareGoodnessOfFit(observed, expected);

      expect(result.isSignificant).toBe(true);
    });

    it('should not flag matching distributions', () => {
      const observed = [50, 50, 50, 50];
      const expected = [50, 50, 50, 50];
      const result = chiSquareGoodnessOfFit(observed, expected);

      expect(result.isSignificant).toBe(false);
    });
  });

  describe('oneWayANOVA', () => {
    it('should detect significant difference between groups', () => {
      const groups = [
        [1, 2, 3, 2, 1],
        [5, 6, 7, 6, 5],
        [10, 11, 12, 11, 10],
      ];
      const result = oneWayANOVA(groups);

      expect(result.isSignificant).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it('should not detect difference for similar groups', () => {
      const groups = [
        [5, 6, 5, 6, 5],
        [5, 5, 6, 5, 6],
        [6, 5, 5, 6, 5],
      ];
      const result = oneWayANOVA(groups);

      expect(result.isSignificant).toBe(false);
    });
  });

  describe('mannWhitneyU', () => {
    it('should detect difference in distributions', () => {
      const group1 = [1, 2, 3, 4, 5];
      const group2 = [6, 7, 8, 9, 10];
      const result = mannWhitneyU(group1, group2);

      expect(result.isSignificant).toBe(true);
    });
  });

  describe('kolmogorovSmirnovTest', () => {
    it('should detect different distributions', () => {
      const sample1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const sample2 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
      const result = kolmogorovSmirnovTest(sample1, sample2);

      expect(result.isSignificant).toBe(true);
      expect(result.statistic).toBeCloseTo(1, 1); // Max difference should be ~1
    });

    it('should not reject similar distributions', () => {
      const sample1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const sample2 = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
      const result = kolmogorovSmirnovTest(sample1, sample2);

      expect(result.statistic).toBeLessThan(0.5);
    });
  });

  describe('confidenceIntervalMean', () => {
    it('should calculate reasonable confidence interval', () => {
      const data = [8, 9, 10, 11, 12, 10, 10, 9, 11, 10];
      const [lower, upper] = confidenceIntervalMean(data, 0.95);

      expect(lower).toBeLessThan(10);
      expect(upper).toBeGreaterThan(10);
      // Mean should be within interval
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(mean).toBeGreaterThan(lower);
      expect(mean).toBeLessThan(upper);
    });
  });

  describe('confidenceIntervalProportion', () => {
    it('should calculate Wilson score interval', () => {
      const [lower, upper] = confidenceIntervalProportion(70, 100, 0.95);

      expect(lower).toBeGreaterThan(0.6);
      expect(upper).toBeLessThan(0.8);
      expect(lower).toBeLessThan(0.7);
      expect(upper).toBeGreaterThan(0.7);
    });

    it('should handle edge cases', () => {
      const [lower, upper] = confidenceIntervalProportion(0, 100, 0.95);
      expect(lower).toBeGreaterThanOrEqual(0);
      expect(upper).toBeLessThanOrEqual(1);
    });
  });
});
