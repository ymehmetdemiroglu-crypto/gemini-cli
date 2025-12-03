/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Statistical distribution functions and utilities
 */

import type { Distribution } from '../types.js';

/**
 * Generate random samples from a normal distribution using Box-Muller transform
 */
export function normalRandom(mean = 0, std = 1): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * std + mean;
}

/**
 * Probability density function for normal distribution
 */
export function normalPDF(x: number, mean = 0, std = 1): number {
  const coefficient = 1 / (std * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mean) / std, 2);
  return coefficient * Math.exp(exponent);
}

/**
 * Cumulative distribution function for normal distribution
 * Uses Abramowitz and Stegun approximation
 */
export function normalCDF(x: number, mean = 0, std = 1): number {
  const z = (x - mean) / std;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/**
 * Inverse normal CDF (quantile function) using rational approximation
 */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  // Rational approximation for central region
  if (p > 0.5) {
    return -normalQuantile(1 - p);
  }

  const t = Math.sqrt(-2 * Math.log(p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t));
}

/**
 * Beta distribution PDF
 */
export function betaPDF(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;
  const B = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta);
  return (Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1)) / B;
}

/**
 * Gamma function approximation using Lanczos approximation
 */
export function gamma(z: number): number {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }

  z -= 1;
  const g = 7;
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  let x = coefficients[0];
  for (let i = 1; i < g + 2; i++) {
    x += coefficients[i] / (z + i);
  }

  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

/**
 * Log gamma function for numerical stability
 */
export function logGamma(z: number): number {
  if (z <= 0) return Infinity;

  const coefficients = [
    76.18009173, -86.50532033, 24.01409824,
    -1.23173957, 1.2086509738e-3, -5.39523938e-6,
  ];

  const x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.00000000019;
  for (let j = 0; j < 6; j++) {
    ser += coefficients[j] / ++y;
  }
  return -tmp + Math.log((2.506628274631 * ser) / x);
}

/**
 * Chi-square CDF using incomplete gamma function
 */
export function chiSquareCDF(x: number, df: number): number {
  if (x <= 0) return 0;
  return incompleteGamma(df / 2, x / 2);
}

/**
 * Incomplete gamma function (regularized)
 */
export function incompleteGamma(a: number, x: number): number {
  if (x < 0 || a <= 0) return 0;

  if (x < a + 1) {
    // Use series representation
    return gammaSeries(a, x);
  } else {
    // Use continued fraction representation
    return 1 - gammaContinuedFraction(a, x);
  }
}

function gammaSeries(a: number, x: number): number {
  const maxIterations = 100;
  const eps = 1e-10;

  let sum = 1 / a;
  let term = 1 / a;

  for (let n = 1; n < maxIterations; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < eps * Math.abs(sum)) break;
  }

  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

function gammaContinuedFraction(a: number, x: number): number {
  const maxIterations = 100;
  const eps = 1e-10;

  let b = x + 1 - a;
  let c = 1 / 1e-30;
  let d = 1 / b;
  let h = d;

  for (let i = 1; i <= maxIterations; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = b + an / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < eps) break;
  }

  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

/**
 * Student's t-distribution CDF
 */
export function tCDF(t: number, df: number): number {
  const x = df / (df + t * t);
  const prob = 0.5 * incompleteBeta(df / 2, 0.5, x);
  return t >= 0 ? 1 - prob : prob;
}

/**
 * Incomplete beta function (regularized)
 */
export function incompleteBeta(a: number, b: number, x: number): number {
  if (x === 0) return 0;
  if (x === 1) return 1;

  const bt =
    Math.exp(
      logGamma(a + b) -
        logGamma(a) -
        logGamma(b) +
        a * Math.log(x) +
        b * Math.log(1 - x)
    );

  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betaContinuedFraction(a, b, x)) / a;
  } else {
    return 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
  }
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const maxIterations = 100;
  const eps = 1e-10;

  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIterations; m++) {
    const m2 = 2 * m;

    // Even step
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;

    // Odd step
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = d * c;
    h *= delta;

    if (Math.abs(delta - 1) < eps) break;
  }

  return h;
}

/**
 * F-distribution CDF
 */
export function fCDF(f: number, df1: number, df2: number): number {
  if (f <= 0) return 0;
  const x = (df1 * f) / (df1 * f + df2);
  return incompleteBeta(df1 / 2, df2 / 2, x);
}

/**
 * Create a distribution object from data
 */
export function createDistributionFromData(data: number[]): Distribution {
  const n = data.length;
  if (n === 0) {
    return { type: 'normal', mean: 0, std: 1 };
  }

  const mean = data.reduce((sum, x) => sum + x, 0) / n;
  const variance =
    data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / (n - 1);
  const std = Math.sqrt(variance);

  return {
    type: 'normal',
    mean,
    std: std || 1,
  };
}

/**
 * Create a categorical distribution from data
 */
export function createCategoricalDistribution(
  data: string[]
): Distribution {
  const counts = new Map<string, number>();
  for (const item of data) {
    counts.set(item, (counts.get(item) || 0) + 1);
  }

  const total = data.length;
  const categories = new Map<string, number>();
  for (const [key, count] of counts) {
    categories.set(key, count / total);
  }

  return {
    type: 'categorical',
    categories,
  };
}

/**
 * Create a histogram distribution from data
 */
export function createHistogramDistribution(
  data: number[],
  numBins = 10
): Distribution {
  if (data.length === 0) {
    return { type: 'histogram', bins: [], frequencies: [] };
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const binWidth = (max - min) / numBins || 1;

  const bins: number[] = [];
  const frequencies: number[] = new Array(numBins).fill(0);

  for (let i = 0; i <= numBins; i++) {
    bins.push(min + i * binWidth);
  }

  for (const value of data) {
    const binIndex = Math.min(
      Math.floor((value - min) / binWidth),
      numBins - 1
    );
    frequencies[binIndex]++;
  }

  // Normalize frequencies
  const total = data.length;
  for (let i = 0; i < frequencies.length; i++) {
    frequencies[i] /= total;
  }

  return {
    type: 'histogram',
    bins,
    frequencies,
  };
}

/**
 * Create a Beta distribution from observed successes/failures
 */
export function createBetaDistribution(
  successes: number,
  failures: number,
  priorAlpha = 1,
  priorBeta = 1
): Distribution {
  return {
    type: 'beta',
    alpha: priorAlpha + successes,
    beta: priorBeta + failures,
  };
}

/**
 * Calculate mean of a distribution
 */
export function distributionMean(dist: Distribution): number {
  switch (dist.type) {
    case 'normal':
      return dist.mean ?? 0;
    case 'beta':
      return (dist.alpha ?? 1) / ((dist.alpha ?? 1) + (dist.beta ?? 1));
    case 'histogram': {
      if (!dist.bins || !dist.frequencies) return 0;
      let sum = 0;
      for (let i = 0; i < dist.frequencies.length; i++) {
        const binCenter = (dist.bins[i] + dist.bins[i + 1]) / 2;
        sum += binCenter * dist.frequencies[i];
      }
      return sum;
    }
    case 'categorical':
      return 0; // Not applicable
    default:
      return 0;
  }
}

/**
 * Calculate variance of a distribution
 */
export function distributionVariance(dist: Distribution): number {
  switch (dist.type) {
    case 'normal':
      return Math.pow(dist.std ?? 1, 2);
    case 'beta': {
      const a = dist.alpha ?? 1;
      const b = dist.beta ?? 1;
      return (a * b) / (Math.pow(a + b, 2) * (a + b + 1));
    }
    default:
      return 0;
  }
}

/**
 * Sample from a distribution
 */
export function sampleFromDistribution(dist: Distribution): number {
  switch (dist.type) {
    case 'normal':
      return normalRandom(dist.mean ?? 0, dist.std ?? 1);
    case 'beta':
      return sampleBeta(dist.alpha ?? 1, dist.beta ?? 1);
    case 'histogram':
      return sampleHistogram(dist);
    default:
      return 0;
  }
}

/**
 * Sample from Beta distribution using gamma sampling
 */
function sampleBeta(alpha: number, beta: number): number {
  const x = sampleGamma(alpha, 1);
  const y = sampleGamma(beta, 1);
  return x / (x + y);
}

/**
 * Sample from Gamma distribution using Marsaglia and Tsang's method
 */
function sampleGamma(shape: number, scale: number): number {
  if (shape < 1) {
    return sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x: number;
    let v: number;

    do {
      x = normalRandom();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.random();

    if (u < 1 - 0.0331 * x * x * x * x) {
      return d * v * scale;
    }

    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
}

/**
 * Sample from histogram distribution
 */
function sampleHistogram(dist: Distribution): number {
  if (!dist.bins || !dist.frequencies || dist.bins.length < 2) {
    return 0;
  }

  const r = Math.random();
  let cumulative = 0;

  for (let i = 0; i < dist.frequencies.length; i++) {
    cumulative += dist.frequencies[i];
    if (r <= cumulative) {
      // Sample uniformly within the bin
      return dist.bins[i] + Math.random() * (dist.bins[i + 1] - dist.bins[i]);
    }
  }

  return dist.bins[dist.bins.length - 1];
}
