/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Statistically-Validated Persona Generation Engine
 *
 * A comprehensive persona generation system that combines:
 * - Bootstrapped sampling for cluster stability
 * - Cluster validity indices (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn)
 * - Monte Carlo validation for robustness testing
 * - Bayesian updating for incremental learning
 * - Hypothesis testing for trait validation
 * - Confidence intervals for all measurements
 *
 * @example
 * ```typescript
 * import { PersonaEngine, generateSampleData } from '@google/persona-engine';
 *
 * const engine = new PersonaEngine({
 *   bootstrapSamples: 100,
 *   monteCarloIterations: 50,
 *   significanceLevel: 0.05,
 * });
 *
 * const data = generateSampleData(500);
 * const result = await engine.generatePersonas(data);
 *
 * console.log(`Generated ${result.personas.length} personas`);
 * for (const persona of result.personas) {
 *   console.log(`${persona.name}: ${persona.overallConfidence * 100}% confidence`);
 * }
 * ```
 */

// Core types
export * from './types.js';

// Statistics module
export {
  // Distributions
  normalRandom,
  normalPDF,
  normalCDF,
  normalQuantile,
  betaPDF,
  gamma,
  logGamma,
  chiSquareCDF,
  tCDF,
  fCDF,
  createDistributionFromData,
  createCategoricalDistribution,
  createHistogramDistribution,
  createBetaDistribution,
  distributionMean,
  distributionVariance,
  sampleFromDistribution,
  // Hypothesis testing
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
  selectHypothesisTest,
} from './statistics/index.js';

// Clustering module
export {
  KMeans,
  buildClusters,
  findOptimalK,
  HDBSCAN,
  buildHDBSCANClusters,
  BootstrappedClustering,
  runBootstrapClustering,
} from './clustering/index.js';

// Validation module
export {
  calculateClusterValidityIndices,
  calculateSilhouetteScore,
  calculateDaviesBouldinIndex,
  calculateCalinskiHarabaszIndex,
  calculateDunnIndex,
  calculateClusterDensity,
  isValidCluster,
  getClusterQualityRating,
  MonteCarloValidator,
  runMonteCarloValidation,
  interpretRobustnessScore,
} from './validation/index.js';

// Bayesian module
export {
  BayesianPersonaUpdater,
  bayesianUpdatePersona,
} from './bayesian/index.js';

// Traits module
export {
  TraitExtractor,
  calculateOverallTraitConfidence,
  formatTrait,
} from './traits/index.js';

// Main engine
export {
  PersonaEngine,
  formatPersona,
  formatGenerationResult,
} from './engine/index.js';
