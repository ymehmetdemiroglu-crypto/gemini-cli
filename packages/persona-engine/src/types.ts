/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Type definitions for the Statistically-Validated Persona Generation Engine
 */

// ============================================================================
// Core Data Types
// ============================================================================

/**
 * A single data point representing user behavior/content
 */
export interface DataPoint {
  id: string;
  embedding: number[];
  metadata: Record<string, unknown>;
  text?: string;
  timestamp?: Date;
  source?: string;
}

/**
 * Trait types that can be measured for personas
 */
export type TraitType =
  | 'topic_frequency'
  | 'sentiment'
  | 'keyword_tfidf'
  | 'behavior'
  | 'posting_time'
  | 'device_usage'
  | 'price_sensitivity'
  | 'custom';

/**
 * A trait with statistical backing
 */
export interface Trait {
  name: string;
  type: TraitType;
  value: number | string | Distribution;
  confidence: number; // 0-1
  pValue?: number;
  sampleSize: number;
  testType?: HypothesisTestType;
}

/**
 * Statistical distribution representation
 */
export interface Distribution {
  type: 'normal' | 'categorical' | 'beta' | 'histogram';
  mean?: number;
  std?: number;
  categories?: Map<string, number>;
  bins?: number[];
  frequencies?: number[];
  alpha?: number; // For Beta distribution
  beta?: number; // For Beta distribution
}

// ============================================================================
// Cluster Types
// ============================================================================

/**
 * A cluster of data points
 */
export interface Cluster {
  id: string;
  centroid: number[];
  members: string[]; // DataPoint IDs
  size: number;
  density?: number;
}

/**
 * Cluster validity indices
 */
export interface ClusterValidityIndices {
  silhouette: number; // -1 to 1 (higher is better)
  daviesBouldin: number; // 0+ (lower is better)
  calinskiHarabasz: number; // 0+ (higher is better)
  dunn: number; // 0+ (higher is better)
}

/**
 * Cluster stability metrics from bootstrap/Monte Carlo
 */
export interface ClusterStability {
  stabilityScore: number; // 0-1
  bootstrapConsistency: number; // 0-1
  confidenceInterval: [number, number];
  appearanceRate: number; // How often this cluster appears across samples
}

// ============================================================================
// Persona Types
// ============================================================================

/**
 * A statistically-validated persona
 */
export interface Persona {
  id: string;
  name: string;
  description: string;
  cluster: Cluster;
  traits: Trait[];
  validityIndices: ClusterValidityIndices;
  stability: ClusterStability;
  robustnessScore: number; // 0-1 from Monte Carlo
  overallConfidence: number; // 0-1
  sampleSize: number;
  lastUpdated: Date;
  version: number;
}

/**
 * Prior distribution for Bayesian updating
 */
export interface Prior {
  alpha: number;
  beta: number;
  observations: number;
}

/**
 * Bayesian persona state for incremental updates
 */
export interface BayesianPersonaState {
  personaId: string;
  priors: Map<string, Prior>; // Trait name -> Prior
  posteriors: Map<string, Distribution>;
  updateCount: number;
  lastUpdate: Date;
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Hypothesis test types supported
 */
export type HypothesisTestType =
  | 'chi_square'
  | 't_test'
  | 'anova'
  | 'mann_whitney'
  | 'kolmogorov_smirnov';

/**
 * Clustering algorithm options
 */
export type ClusteringAlgorithm = 'kmeans' | 'hdbscan' | 'agglomerative';

/**
 * Configuration for the persona engine
 */
export interface PersonaEngineConfig {
  // Clustering
  clusteringAlgorithm: ClusteringAlgorithm;
  minClusterSize: number;
  maxClusters: number;

  // Bootstrap sampling
  bootstrapSamples: number;
  sampleSize: number;
  minSampleSize: number;

  // Monte Carlo validation
  monteCarloIterations: number;
  perturbationMagnitude: number;

  // Statistical thresholds
  significanceLevel: number; // Default 0.05
  minConfidence: number;
  minStabilityScore: number;

  // Bayesian updating
  enableBayesianUpdates: boolean;
  priorStrength: number;

  // Output
  minPersonaSize: number;
  maxTraitsPerPersona: number;
}

/**
 * Result of a single bootstrap iteration
 */
export interface BootstrapResult {
  clusters: Cluster[];
  validityIndices: ClusterValidityIndices;
  sampleIndices: number[];
}

/**
 * Result of Monte Carlo validation
 */
export interface MonteCarloResult {
  robustnessScore: number;
  membershipMatrix: number[][]; // Probability of each point being in each cluster
  clusterPersistence: number[];
  convergenceIterations: number;
}

/**
 * Result of hypothesis testing
 */
export interface HypothesisTestResult {
  testType: HypothesisTestType;
  statistic: number;
  pValue: number;
  degreesOfFreedom?: number;
  effectSize?: number;
  isSignificant: boolean;
  confidenceInterval?: [number, number];
}

/**
 * Engine output - full persona generation result
 */
export interface PersonaGenerationResult {
  personas: Persona[];
  metadata: {
    totalDataPoints: number;
    processingTime: number;
    bootstrapIterations: number;
    monteCarloIterations: number;
    averageClusterValidity: ClusterValidityIndices;
    overallConfidence: number;
  };
  warnings: string[];
}
