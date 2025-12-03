/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hypothesis testing functions for statistical validation of persona traits
 */

import type { HypothesisTestResult, HypothesisTestType } from '../types.js';
import { chiSquareCDF, tCDF, fCDF, normalCDF } from './distributions.js';

/**
 * Calculate basic descriptive statistics
 */
export interface DescriptiveStats {
  mean: number;
  variance: number;
  std: number;
  n: number;
  min: number;
  max: number;
}

export function calculateStats(data: number[]): DescriptiveStats {
  const n = data.length;
  if (n === 0) {
    return { mean: 0, variance: 0, std: 0, n: 0, min: 0, max: 0 };
  }

  const mean = data.reduce((sum, x) => sum + x, 0) / n;
  const variance =
    n > 1 ? data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);

  return {
    mean,
    variance,
    std,
    n,
    min: Math.min(...data),
    max: Math.max(...data),
  };
}

/**
 * One-sample t-test: tests if sample mean differs from hypothesized value
 */
export function oneSampleTTest(
  data: number[],
  hypothesizedMean: number,
  alpha = 0.05
): HypothesisTestResult {
  const stats = calculateStats(data);
  const n = stats.n;

  if (n < 2) {
    return {
      testType: 't_test',
      statistic: 0,
      pValue: 1,
      degreesOfFreedom: 0,
      isSignificant: false,
    };
  }

  const se = stats.std / Math.sqrt(n);
  const t = (stats.mean - hypothesizedMean) / se;
  const df = n - 1;

  // Two-tailed p-value
  const pValue = 2 * (1 - tCDF(Math.abs(t), df));

  // Cohen's d effect size
  const effectSize = (stats.mean - hypothesizedMean) / stats.std;

  return {
    testType: 't_test',
    statistic: t,
    pValue,
    degreesOfFreedom: df,
    effectSize,
    isSignificant: pValue < alpha,
  };
}

/**
 * Two-sample independent t-test: tests if two group means differ
 */
export function twoSampleTTest(
  group1: number[],
  group2: number[],
  alpha = 0.05
): HypothesisTestResult {
  const stats1 = calculateStats(group1);
  const stats2 = calculateStats(group2);

  if (stats1.n < 2 || stats2.n < 2) {
    return {
      testType: 't_test',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Welch's t-test (unequal variances)
  const se1 = stats1.variance / stats1.n;
  const se2 = stats2.variance / stats2.n;
  const se = Math.sqrt(se1 + se2);

  const t = (stats1.mean - stats2.mean) / se;

  // Welch-Satterthwaite degrees of freedom
  const df =
    Math.pow(se1 + se2, 2) /
    (Math.pow(se1, 2) / (stats1.n - 1) + Math.pow(se2, 2) / (stats2.n - 1));

  const pValue = 2 * (1 - tCDF(Math.abs(t), df));

  // Cohen's d effect size
  const pooledStd = Math.sqrt(
    ((stats1.n - 1) * stats1.variance + (stats2.n - 1) * stats2.variance) /
      (stats1.n + stats2.n - 2)
  );
  const effectSize = (stats1.mean - stats2.mean) / pooledStd;

  return {
    testType: 't_test',
    statistic: t,
    pValue,
    degreesOfFreedom: df,
    effectSize,
    isSignificant: pValue < alpha,
  };
}

/**
 * Chi-square test for independence: tests if two categorical variables are independent
 */
export function chiSquareTest(
  observed: number[][],
  alpha = 0.05
): HypothesisTestResult {
  const rows = observed.length;
  if (rows === 0) {
    return {
      testType: 'chi_square',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }
  const cols = observed[0].length;

  // Calculate row and column totals
  const rowTotals: number[] = [];
  const colTotals: number[] = new Array(cols).fill(0);
  let grandTotal = 0;

  for (let i = 0; i < rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < cols; j++) {
      rowSum += observed[i][j];
      colTotals[j] += observed[i][j];
    }
    rowTotals.push(rowSum);
    grandTotal += rowSum;
  }

  if (grandTotal === 0) {
    return {
      testType: 'chi_square',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Calculate expected frequencies and chi-square statistic
  let chiSquare = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const expected = (rowTotals[i] * colTotals[j]) / grandTotal;
      if (expected > 0) {
        chiSquare += Math.pow(observed[i][j] - expected, 2) / expected;
      }
    }
  }

  const df = (rows - 1) * (cols - 1);
  const pValue = 1 - chiSquareCDF(chiSquare, df);

  // Cramér's V effect size
  const minDim = Math.min(rows - 1, cols - 1);
  const cramersV =
    minDim > 0 ? Math.sqrt(chiSquare / (grandTotal * minDim)) : 0;

  return {
    testType: 'chi_square',
    statistic: chiSquare,
    pValue,
    degreesOfFreedom: df,
    effectSize: cramersV,
    isSignificant: pValue < alpha,
  };
}

/**
 * Chi-square goodness of fit test: tests if observed frequencies match expected
 */
export function chiSquareGoodnessOfFit(
  observed: number[],
  expected: number[],
  alpha = 0.05
): HypothesisTestResult {
  if (observed.length !== expected.length || observed.length === 0) {
    return {
      testType: 'chi_square',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  let chiSquare = 0;
  for (let i = 0; i < observed.length; i++) {
    if (expected[i] > 0) {
      chiSquare += Math.pow(observed[i] - expected[i], 2) / expected[i];
    }
  }

  const df = observed.length - 1;
  const pValue = 1 - chiSquareCDF(chiSquare, df);

  return {
    testType: 'chi_square',
    statistic: chiSquare,
    pValue,
    degreesOfFreedom: df,
    isSignificant: pValue < alpha,
  };
}

/**
 * One-way ANOVA: tests if means of multiple groups differ
 */
export function oneWayANOVA(
  groups: number[][],
  alpha = 0.05
): HypothesisTestResult {
  const k = groups.length;
  if (k < 2) {
    return {
      testType: 'anova',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Calculate group means and sizes
  const groupStats = groups.map((g) => calculateStats(g));
  const N = groupStats.reduce((sum, s) => sum + s.n, 0);

  if (N <= k) {
    return {
      testType: 'anova',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Grand mean
  const grandMean =
    groupStats.reduce((sum, s) => sum + s.mean * s.n, 0) / N;

  // Between-group sum of squares (SSB)
  const SSB = groupStats.reduce(
    (sum, s) => sum + s.n * Math.pow(s.mean - grandMean, 2),
    0
  );

  // Within-group sum of squares (SSW)
  const SSW = groupStats.reduce((sum, s) => sum + (s.n - 1) * s.variance, 0);

  // Mean squares
  const dfBetween = k - 1;
  const dfWithin = N - k;
  const MSB = SSB / dfBetween;
  const MSW = SSW / dfWithin;

  // F-statistic
  const F = MSW > 0 ? MSB / MSW : 0;
  const pValue = 1 - fCDF(F, dfBetween, dfWithin);

  // Effect size (eta-squared)
  const SST = SSB + SSW;
  const etaSquared = SST > 0 ? SSB / SST : 0;

  return {
    testType: 'anova',
    statistic: F,
    pValue,
    degreesOfFreedom: dfBetween,
    effectSize: etaSquared,
    isSignificant: pValue < alpha,
  };
}

/**
 * Mann-Whitney U test: non-parametric test for comparing two groups
 */
export function mannWhitneyU(
  group1: number[],
  group2: number[],
  alpha = 0.05
): HypothesisTestResult {
  const n1 = group1.length;
  const n2 = group2.length;

  if (n1 === 0 || n2 === 0) {
    return {
      testType: 'mann_whitney',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Combine and rank
  const combined: Array<{ value: number; group: number }> = [
    ...group1.map((v) => ({ value: v, group: 1 })),
    ...group2.map((v) => ({ value: v, group: 2 })),
  ];
  combined.sort((a, b) => a.value - b.value);

  // Assign ranks (handling ties with average rank)
  const ranks = assignRanks(combined.map((c) => c.value));

  // Calculate rank sums
  let R1 = 0;
  let R2 = 0;
  for (let i = 0; i < combined.length; i++) {
    if (combined[i].group === 1) {
      R1 += ranks[i];
    } else {
      R2 += ranks[i];
    }
  }

  // Calculate U statistics
  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const U2 = R2 - (n2 * (n2 + 1)) / 2;
  const U = Math.min(U1, U2);

  // Normal approximation for large samples
  const muU = (n1 * n2) / 2;
  const sigmaU = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
  const z = (U - muU) / sigmaU;
  const pValue = 2 * normalCDF(-Math.abs(z));

  // Effect size (rank-biserial correlation)
  const effectSize = 1 - (2 * U) / (n1 * n2);

  return {
    testType: 'mann_whitney',
    statistic: U,
    pValue,
    effectSize,
    isSignificant: pValue < alpha,
  };
}

/**
 * Kolmogorov-Smirnov test: tests if two samples come from same distribution
 */
export function kolmogorovSmirnovTest(
  sample1: number[],
  sample2: number[],
  alpha = 0.05
): HypothesisTestResult {
  const n1 = sample1.length;
  const n2 = sample2.length;

  if (n1 === 0 || n2 === 0) {
    return {
      testType: 'kolmogorov_smirnov',
      statistic: 0,
      pValue: 1,
      isSignificant: false,
    };
  }

  // Sort both samples
  const sorted1 = [...sample1].sort((a, b) => a - b);
  const sorted2 = [...sample2].sort((a, b) => a - b);

  // Calculate maximum distance between ECDFs
  let D = 0;
  let i = 0;
  let j = 0;

  while (i < n1 || j < n2) {
    const val1 = i < n1 ? sorted1[i] : Infinity;
    const val2 = j < n2 ? sorted2[j] : Infinity;

    if (val1 <= val2 && i < n1) i++;
    if (val2 <= val1 && j < n2) j++;

    const ecdf1 = i / n1;
    const ecdf2 = j / n2;
    D = Math.max(D, Math.abs(ecdf1 - ecdf2));
  }

  // Calculate p-value using asymptotic distribution
  const en = Math.sqrt((n1 * n2) / (n1 + n2));
  const lambda = (en + 0.12 + 0.11 / en) * D;

  // Kolmogorov distribution approximation
  let pValue = 0;
  for (let k = 1; k <= 100; k++) {
    pValue +=
      2 * Math.pow(-1, k + 1) * Math.exp(-2 * k * k * lambda * lambda);
  }
  pValue = Math.max(0, Math.min(1, pValue));

  return {
    testType: 'kolmogorov_smirnov',
    statistic: D,
    pValue,
    isSignificant: pValue < alpha,
  };
}

/**
 * Assign ranks to values, handling ties with average rank
 */
function assignRanks(values: number[]): number[] {
  const n = values.length;
  const ranks = new Array(n);
  const sorted = values
    .map((v, i) => ({ value: v, index: i }))
    .sort((a, b) => a.value - b.value);

  let i = 0;
  while (i < n) {
    let j = i;
    // Find all tied values
    while (j < n && sorted[j].value === sorted[i].value) {
      j++;
    }
    // Average rank for tied values
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[sorted[k].index] = avgRank;
    }
    i = j;
  }

  return ranks;
}

/**
 * Calculate confidence interval for a mean
 */
export function confidenceIntervalMean(
  data: number[],
  confidenceLevel = 0.95
): [number, number] {
  const stats = calculateStats(data);
  if (stats.n < 2) {
    return [stats.mean, stats.mean];
  }

  const alpha = 1 - confidenceLevel;
  // Use t-distribution critical value (approximation)
  const tCritical = tCriticalValue(alpha / 2, stats.n - 1);
  const marginOfError = (tCritical * stats.std) / Math.sqrt(stats.n);

  return [stats.mean - marginOfError, stats.mean + marginOfError];
}

/**
 * Calculate confidence interval for a proportion
 */
export function confidenceIntervalProportion(
  successes: number,
  total: number,
  confidenceLevel = 0.95
): [number, number] {
  if (total === 0) return [0, 1];

  const p = successes / total;
  const alpha = 1 - confidenceLevel;
  const z = -1 * normalQuantileApprox(alpha / 2);

  // Wilson score interval (better for small samples)
  const denominator = 1 + (z * z) / total;
  const center = p + (z * z) / (2 * total);
  const marginOfError =
    (z * Math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total))) /
    denominator;

  const lower = Math.max(0, (center - marginOfError) / denominator);
  const upper = Math.min(1, (center + marginOfError) / denominator);

  return [lower, upper];
}

/**
 * Approximate t critical value
 */
function tCriticalValue(alpha: number, df: number): number {
  // Use normal approximation for large df
  if (df > 120) {
    return -normalQuantileApprox(alpha);
  }

  // Simple approximation for smaller df
  const a = 1 - alpha;
  const za = -normalQuantileApprox(1 - a);

  // Cornish-Fisher expansion
  const g1 = (za * za * za + za) / 4;
  const g2 = ((5 * za * za * za * za * za + 16 * za * za * za + 3 * za)) / 96;
  const g3 =
    ((3 * za * za * za * za * za * za * za + 19 * za * za * za * za * za + 17 * za * za * za - 15 * za)) / 384;

  return za + g1 / df + g2 / (df * df) + g3 / (df * df * df);
}

/**
 * Simple normal quantile approximation
 */
function normalQuantileApprox(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  if (p > 0.5) {
    return -normalQuantileApprox(1 - p);
  }

  const t = Math.sqrt(-2 * Math.log(p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return -(
    t -
    (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
  );
}

/**
 * Select appropriate hypothesis test based on data characteristics
 */
export function selectHypothesisTest(
  data1: number[] | string[],
  data2?: number[] | string[],
  groups?: number[][]
): HypothesisTestType {
  // If we have groups for ANOVA
  if (groups && groups.length > 2) {
    return 'anova';
  }

  // Check if data is categorical
  const isCategorical1 = typeof data1[0] === 'string';
  const isCategorical2 = data2 && typeof data2[0] === 'string';

  if (isCategorical1 || isCategorical2) {
    return 'chi_square';
  }

  // For two numeric samples
  if (data2 && data1.length > 0 && data2.length > 0) {
    // Check for normality (simplified - use Shapiro-Wilk in production)
    const combined = [...(data1 as number[]), ...(data2 as number[])];
    if (isApproximatelyNormal(combined as number[])) {
      return 't_test';
    }
    return 'mann_whitney';
  }

  // Default for single sample
  return 't_test';
}

/**
 * Simple normality check using skewness and kurtosis
 */
function isApproximatelyNormal(data: number[]): boolean {
  if (data.length < 20) return false;

  const stats = calculateStats(data);
  const n = stats.n;

  // Calculate skewness
  let m3 = 0;
  let m4 = 0;
  for (const x of data) {
    const diff = x - stats.mean;
    m3 += Math.pow(diff, 3);
    m4 += Math.pow(diff, 4);
  }
  m3 /= n;
  m4 /= n;

  const skewness = m3 / Math.pow(stats.std, 3);
  const kurtosis = m4 / Math.pow(stats.std, 4) - 3; // Excess kurtosis

  // Normal distribution has skewness ≈ 0 and kurtosis ≈ 0
  return Math.abs(skewness) < 2 && Math.abs(kurtosis) < 7;
}
