/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Main Persona Engine Orchestrator
 * Ties together all statistical components for persona generation
 */

import type {
  DataPoint,
  Persona,
  PersonaEngineConfig,
  PersonaGenerationResult,
  Cluster,
  ClusterValidityIndices,
  ClusterStability,
} from '../types.js';
import { BootstrappedClustering } from '../clustering/bootstrap.js';
import { calculateClusterValidityIndices } from '../validation/cluster-validity.js';
import {
  MonteCarloValidator,
  interpretRobustnessScore,
} from '../validation/monte-carlo.js';
import { BayesianPersonaUpdater } from '../bayesian/bayesian-updater.js';
import {
  TraitExtractor,
  calculateOverallTraitConfidence,
} from '../traits/trait-extractor.js';

/**
 * Default configuration for the persona engine
 */
const DEFAULT_CONFIG: PersonaEngineConfig = {
  // Clustering
  clusteringAlgorithm: 'kmeans',
  minClusterSize: 10,
  maxClusters: 10,

  // Bootstrap sampling
  bootstrapSamples: 100,
  sampleSize: 0.8,
  minSampleSize: 50,

  // Monte Carlo validation
  monteCarloIterations: 50,
  perturbationMagnitude: 0.1,

  // Statistical thresholds
  significanceLevel: 0.05,
  minConfidence: 0.5,
  minStabilityScore: 0.4,

  // Bayesian updating
  enableBayesianUpdates: true,
  priorStrength: 10,

  // Output
  minPersonaSize: 10,
  maxTraitsPerPersona: 15,
};

/**
 * Statistically-Validated Persona Generation Engine
 *
 * This engine combines:
 * - Bootstrapped sampling for cluster stability
 * - Cluster validity indices (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn)
 * - Monte Carlo validation for robustness
 * - Bayesian updating for incremental learning
 * - Hypothesis testing for trait validation
 * - Confidence intervals for all measurements
 */
export class PersonaEngine {
  private config: PersonaEngineConfig;
  private bayesianUpdater: BayesianPersonaUpdater;
  private traitExtractor: TraitExtractor;
  private personas: Map<string, Persona> = new Map();
  private lastGenerationResult: PersonaGenerationResult | null = null;

  constructor(config: Partial<PersonaEngineConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.bayesianUpdater = new BayesianPersonaUpdater({
      priorStrength: this.config.priorStrength,
    });
    this.traitExtractor = new TraitExtractor({
      significanceLevel: this.config.significanceLevel,
      minConfidence: this.config.minConfidence,
    });
  }

  /**
   * Generate personas from raw data points
   *
   * The full pipeline:
   * 1. Embed data (assumed to be pre-embedded)
   * 2. Bootstrapped clustering for stability
   * 3. Cluster validity indices calculation
   * 4. Monte Carlo validation for robustness
   * 5. Trait extraction with hypothesis testing
   * 6. Persona synthesis with confidence intervals
   */
  async generatePersonas(dataPoints: DataPoint[]): Promise<PersonaGenerationResult> {
    const startTime = Date.now();
    const warnings: string[] = [];

    // Validation
    if (dataPoints.length < this.config.minSampleSize) {
      warnings.push(
        `Sample size (${dataPoints.length}) is below minimum (${this.config.minSampleSize}). Results may be unreliable.`
      );
    }

    // Step 1: Bootstrapped Clustering
    console.log('🔄 Running bootstrapped clustering...');
    const bootstrap = new BootstrappedClustering({
      numSamples: this.config.bootstrapSamples,
      sampleSize: this.config.sampleSize,
      sampleAsFraction: true,
      algorithm: this.config.clusteringAlgorithm,
      minClusterSize: this.config.minClusterSize,
    });

    bootstrap.run(dataPoints);
    const stableClusters = bootstrap.getStableClusters(this.config.minStabilityScore);
    const stabilityMap = bootstrap.calculateStability();

    if (stableClusters.length === 0) {
      warnings.push('No stable clusters found. Consider adjusting parameters.');
      return {
        personas: [],
        metadata: {
          totalDataPoints: dataPoints.length,
          processingTime: Date.now() - startTime,
          bootstrapIterations: this.config.bootstrapSamples,
          monteCarloIterations: 0,
          averageClusterValidity: {
            silhouette: 0,
            daviesBouldin: Infinity,
            calinskiHarabasz: 0,
            dunn: 0,
          },
          overallConfidence: 0,
        },
        warnings,
      };
    }

    // Step 2: Calculate Cluster Validity Indices
    console.log('📊 Calculating cluster validity indices...');
    const validityIndices = calculateClusterValidityIndices(
      dataPoints,
      stableClusters
    );

    // Step 3: Monte Carlo Validation
    console.log('🎲 Running Monte Carlo validation...');
    const monteCarlo = new MonteCarloValidator({
      iterations: this.config.monteCarloIterations,
      perturbationMagnitude: this.config.perturbationMagnitude,
      algorithm: this.config.clusteringAlgorithm,
      minClusterSize: this.config.minClusterSize,
    });

    // Run validation (result is accessed via getClusterRobustness)
    monteCarlo.validate(dataPoints, stableClusters);
    const clusterRobustness = monteCarlo.getClusterRobustness(
      stableClusters,
      dataPoints
    );

    // Step 4: Extract Traits and Build Personas
    console.log('🧬 Extracting traits and building personas...');
    const globalStats = this.calculateGlobalStats(dataPoints);
    const personas: Persona[] = [];

    for (let i = 0; i < stableClusters.length; i++) {
      const cluster = stableClusters[i];
      const clusterDataPoints = this.getClusterDataPoints(dataPoints, cluster);

      // Skip small clusters
      if (clusterDataPoints.length < this.config.minPersonaSize) {
        warnings.push(
          `Cluster ${cluster.id} has only ${clusterDataPoints.length} members, below minimum ${this.config.minPersonaSize}`
        );
        continue;
      }

      // Extract traits with statistical validation
      const traits = this.traitExtractor.extractTraits(
        clusterDataPoints,
        globalStats
      );

      // Get stability and robustness scores
      const stabilityKey = this.getClusterSignature(cluster);
      const stability = stabilityMap.get(stabilityKey) || {
        stabilityScore: 0,
        bootstrapConsistency: 0,
        confidenceInterval: [0, 1] as [number, number],
        appearanceRate: 0,
      };

      const robustnessScore = clusterRobustness.get(cluster.id) || 0;

      // Calculate overall confidence
      const traitConfidence = calculateOverallTraitConfidence(traits);
      const overallConfidence = this.calculateOverallConfidence(
        validityIndices,
        stability,
        robustnessScore,
        traitConfidence
      );

      // Generate persona name and description
      const { name, description } = this.generatePersonaDescription(
        traits,
        clusterDataPoints.length
      );

      const persona: Persona = {
        id: `persona_${i + 1}`,
        name,
        description,
        cluster,
        traits: traits.slice(0, this.config.maxTraitsPerPersona),
        validityIndices,
        stability,
        robustnessScore,
        overallConfidence,
        sampleSize: clusterDataPoints.length,
        lastUpdated: new Date(),
        version: 1,
      };

      personas.push(persona);
      this.personas.set(persona.id, persona);

      // Initialize Bayesian state if enabled
      if (this.config.enableBayesianUpdates) {
        this.bayesianUpdater.initializePersona(persona);
      }
    }

    // Sort personas by confidence
    personas.sort((a, b) => b.overallConfidence - a.overallConfidence);

    const result: PersonaGenerationResult = {
      personas,
      metadata: {
        totalDataPoints: dataPoints.length,
        processingTime: Date.now() - startTime,
        bootstrapIterations: this.config.bootstrapSamples,
        monteCarloIterations: this.config.monteCarloIterations,
        averageClusterValidity: validityIndices,
        overallConfidence:
          personas.length > 0
            ? personas.reduce((sum, p) => sum + p.overallConfidence, 0) /
              personas.length
            : 0,
      },
      warnings,
    };

    this.lastGenerationResult = result;
    console.log(`✅ Generated ${personas.length} statistically-validated personas`);

    return result;
  }

  /**
   * Update personas with new data using Bayesian inference
   */
  updateWithNewData(
    newDataPoints: DataPoint[],
    traitExtractor?: (points: DataPoint[]) => Map<string, number | string[]>
  ): Persona[] {
    if (!this.config.enableBayesianUpdates) {
      throw new Error('Bayesian updates are not enabled');
    }

    const defaultExtractor = (points: DataPoint[]) => {
      const traits = new Map<string, number | string[]>();

      // Extract numeric traits
      const sentiments = points
        .map((p) => p.metadata?.sentiment as number)
        .filter((s) => s !== undefined);
      if (sentiments.length > 0) {
        traits.set(
          'sentiment',
          sentiments.reduce((a, b) => a + b, 0) / sentiments.length
        );
      }

      // Extract categorical traits
      const topics: string[] = [];
      for (const point of points) {
        const t = point.metadata?.topics as string[] | undefined;
        if (t) topics.push(...t);
      }
      if (topics.length > 0) {
        traits.set('topics', topics);
      }

      return traits;
    };

    const extractor = traitExtractor || defaultExtractor;
    const updatedPersonas: Persona[] = [];

    for (const persona of this.personas.values()) {
      const relevantPoints = this.assignPointsToPersona(newDataPoints, persona);
      if (relevantPoints.length > 0) {
        const updated = this.bayesianUpdater.updatePersona(
          persona,
          relevantPoints,
          extractor
        );
        this.personas.set(updated.id, updated);
        updatedPersonas.push(updated);
      }
    }

    return updatedPersonas;
  }

  /**
   * Get persona by ID
   */
  getPersona(id: string): Persona | undefined {
    return this.personas.get(id);
  }

  /**
   * Get all personas
   */
  getAllPersonas(): Persona[] {
    return Array.from(this.personas.values());
  }

  /**
   * Get last generation result
   */
  getLastResult(): PersonaGenerationResult | null {
    return this.lastGenerationResult;
  }

  /**
   * Calculate global statistics for comparison
   */
  private calculateGlobalStats(dataPoints: DataPoint[]): {
    topicFrequencies: Map<string, number>;
    avgSentiment: number;
    avgPostingHour: number;
  } {
    const topicCounts = new Map<string, number>();
    let totalTopics = 0;
    const sentiments: number[] = [];
    const hours: number[] = [];

    for (const point of dataPoints) {
      const topics = point.metadata?.topics as string[] | undefined;
      if (topics) {
        for (const topic of topics) {
          topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
          totalTopics++;
        }
      }

      const sentiment = point.metadata?.sentiment as number | undefined;
      if (sentiment !== undefined) sentiments.push(sentiment);

      if (point.timestamp) {
        hours.push(new Date(point.timestamp).getHours());
      }
    }

    const topicFrequencies = new Map<string, number>();
    for (const [topic, count] of topicCounts) {
      topicFrequencies.set(topic, count / totalTopics);
    }

    return {
      topicFrequencies,
      avgSentiment:
        sentiments.length > 0
          ? sentiments.reduce((a, b) => a + b, 0) / sentiments.length
          : 0,
      avgPostingHour:
        hours.length > 0
          ? hours.reduce((a, b) => a + b, 0) / hours.length
          : 12,
    };
  }

  /**
   * Get data points belonging to a cluster
   */
  private getClusterDataPoints(
    dataPoints: DataPoint[],
    cluster: Cluster
  ): DataPoint[] {
    const memberSet = new Set(cluster.members);
    return dataPoints.filter((p) => memberSet.has(p.id));
  }

  /**
   * Assign new data points to personas based on embedding similarity
   */
  private assignPointsToPersona(
    dataPoints: DataPoint[],
    persona: Persona
  ): DataPoint[] {
    const threshold = 0.8; // Cosine similarity threshold
    const assigned: DataPoint[] = [];

    for (const point of dataPoints) {
      const similarity = this.cosineSimilarity(
        point.embedding,
        persona.cluster.centroid
      );
      if (similarity >= threshold) {
        assigned.push(point);
      }
    }

    return assigned;
  }

  /**
   * Calculate overall confidence score
   */
  private calculateOverallConfidence(
    validity: ClusterValidityIndices,
    stability: ClusterStability,
    robustness: number,
    traitConfidence: number
  ): number {
    // Normalize validity indices to [0, 1]
    const silhouetteScore = (validity.silhouette + 1) / 2; // -1 to 1 -> 0 to 1
    const dbScore = Math.max(0, 1 - validity.daviesBouldin / 3); // Lower is better
    const validityScore = (silhouetteScore + dbScore) / 2;

    // Weighted combination
    const weights = {
      validity: 0.25,
      stability: 0.25,
      robustness: 0.25,
      traits: 0.25,
    };

    return (
      validityScore * weights.validity +
      stability.stabilityScore * weights.stability +
      robustness * weights.robustness +
      traitConfidence * weights.traits
    );
  }

  /**
   * Generate persona name and description from traits
   */
  private generatePersonaDescription(
    traits: ReturnType<TraitExtractor['extractTraits']>,
    size: number
  ): { name: string; description: string } {
    // Extract key characteristics
    const topTopics = traits
      .filter((t) => t.type === 'topic_frequency')
      .slice(0, 2)
      .map((t) => t.name.replace('topic_', ''));

    const sentiment = traits.find((t) => t.type === 'sentiment');
    const time = traits.find((t) => t.type === 'posting_time');
    const price = traits.find((t) => t.type === 'price_sensitivity');

    // Generate name
    let name = 'Engaged User';
    if (topTopics.length > 0) {
      name = `${this.capitalize(topTopics[0])} Enthusiast`;
    }
    if (price) {
      const priceLabel = price.name.replace('price_', '');
      if (priceLabel === 'budget_conscious') name = `Budget-Savvy ${name}`;
      else if (priceLabel === 'premium_buyer') name = `Premium ${name}`;
    }

    // Generate description
    const descParts: string[] = [];
    descParts.push(`A persona representing ${size} data points.`);

    if (topTopics.length > 0) {
      descParts.push(`Primarily interested in ${topTopics.join(' and ')}.`);
    }

    if (sentiment) {
      const sentimentLabel = sentiment.name.replace('sentiment_', '');
      descParts.push(`Generally expresses ${sentimentLabel} sentiment.`);
    }

    if (time) {
      const timeLabel = time.name.replace('posting_time_', '');
      descParts.push(`Most active during ${timeLabel} hours.`);
    }

    return {
      name,
      description: descParts.join(' '),
    };
  }

  /**
   * Create cluster signature for stability matching
   */
  private getClusterSignature(cluster: Cluster): string {
    const roundedCentroid = cluster.centroid
      .slice(0, 5)
      .map((v) => Math.round(v * 100) / 100);
    return `c_${roundedCentroid.join('_')}_s${cluster.size}`;
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * (b[i] || 0);
      normA += a[i] * a[i];
      normB += (b[i] || 0) * (b[i] || 0);
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dotProduct / denom : 0;
  }

  private capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
}

/**
 * Format a persona for display
 */
export function formatPersona(persona: Persona): string {
  const lines: string[] = [];

  lines.push(`═══════════════════════════════════════════════════════════`);
  lines.push(`📊 ${persona.name}`);
  lines.push(`   ID: ${persona.id} | Version: ${persona.version}`);
  lines.push(`   Overall Confidence: ${(persona.overallConfidence * 100).toFixed(1)}%`);
  lines.push(`───────────────────────────────────────────────────────────`);
  lines.push(`📝 ${persona.description}`);
  lines.push(`───────────────────────────────────────────────────────────`);
  lines.push(`📈 Statistical Validation:`);
  lines.push(`   • Sample Size: ${persona.sampleSize}`);
  lines.push(`   • Stability Score: ${(persona.stability.stabilityScore * 100).toFixed(1)}%`);
  lines.push(`   • Robustness Score: ${(persona.robustnessScore * 100).toFixed(1)}% (${interpretRobustnessScore(persona.robustnessScore).rating})`);
  lines.push(`   • Silhouette: ${persona.validityIndices.silhouette.toFixed(3)}`);
  lines.push(`   • Davies-Bouldin: ${persona.validityIndices.daviesBouldin.toFixed(3)}`);
  lines.push(`───────────────────────────────────────────────────────────`);
  lines.push(`🏷️  Traits (with confidence):`);

  for (const trait of persona.traits.slice(0, 10)) {
    const conf = (trait.confidence * 100).toFixed(0);
    const pVal = trait.pValue ? ` (p=${trait.pValue.toFixed(4)})` : '';
    lines.push(`   • ${trait.name}: ${conf}% confidence${pVal}`);
  }

  lines.push(`═══════════════════════════════════════════════════════════`);

  return lines.join('\n');
}

/**
 * Format full generation result
 */
export function formatGenerationResult(result: PersonaGenerationResult): string {
  const lines: string[] = [];

  lines.push(`\n🔬 PERSONA GENERATION REPORT`);
  lines.push(`════════════════════════════════════════════════════════════`);
  lines.push(`📊 Metadata:`);
  lines.push(`   • Total Data Points: ${result.metadata.totalDataPoints}`);
  lines.push(`   • Processing Time: ${result.metadata.processingTime}ms`);
  lines.push(`   • Bootstrap Iterations: ${result.metadata.bootstrapIterations}`);
  lines.push(`   • Monte Carlo Iterations: ${result.metadata.monteCarloIterations}`);
  lines.push(`   • Overall Confidence: ${(result.metadata.overallConfidence * 100).toFixed(1)}%`);
  lines.push(`   • Personas Generated: ${result.personas.length}`);

  if (result.warnings.length > 0) {
    lines.push(`\n⚠️  Warnings:`);
    for (const warning of result.warnings) {
      lines.push(`   • ${warning}`);
    }
  }

  lines.push(`\n📈 Cluster Validity (Average):`);
  lines.push(`   • Silhouette Score: ${result.metadata.averageClusterValidity.silhouette.toFixed(3)}`);
  lines.push(`   • Davies-Bouldin Index: ${result.metadata.averageClusterValidity.daviesBouldin.toFixed(3)}`);
  lines.push(`   • Calinski-Harabasz Index: ${result.metadata.averageClusterValidity.calinskiHarabasz.toFixed(3)}`);
  lines.push(`   • Dunn Index: ${result.metadata.averageClusterValidity.dunn.toFixed(3)}`);

  lines.push(`\n════════════════════════════════════════════════════════════`);
  lines.push(`\n🎭 GENERATED PERSONAS:\n`);

  for (const persona of result.personas) {
    lines.push(formatPersona(persona));
    lines.push('');
  }

  return lines.join('\n');
}
