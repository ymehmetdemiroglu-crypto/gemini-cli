/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Bayesian Persona Updating
 * Incrementally update personas with new data using Bayesian inference
 */

import type {
  Persona,
  Trait,
  Distribution,
  DataPoint,
  BayesianPersonaState,
  Prior,
} from '../types.js';
import {
  createBetaDistribution,
  distributionMean,
  distributionVariance,
} from '../statistics/distributions.js';

/**
 * Configuration for Bayesian updating
 */
export interface BayesianConfig {
  priorStrength: number; // How strongly to weight prior beliefs (1-100)
  learningRate: number; // How quickly to update (0-1)
  minObservations: number; // Minimum observations before updating
  decayFactor: number; // Exponential decay for old observations (0-1)
}

const DEFAULT_BAYESIAN_CONFIG: BayesianConfig = {
  priorStrength: 10,
  learningRate: 0.3,
  minObservations: 5,
  decayFactor: 0.95,
};

/**
 * Bayesian updater for persona traits
 */
export class BayesianPersonaUpdater {
  private config: BayesianConfig;
  private states: Map<string, BayesianPersonaState> = new Map();

  constructor(config: Partial<BayesianConfig> = {}) {
    this.config = { ...DEFAULT_BAYESIAN_CONFIG, ...config };
  }

  /**
   * Initialize state for a persona
   */
  initializePersona(persona: Persona): void {
    const priors = new Map<string, Prior>();
    const posteriors = new Map<string, Distribution>();

    // Convert each trait to a prior distribution
    for (const trait of persona.traits) {
      const prior = this.traitToPrior(trait);
      priors.set(trait.name, prior);

      // Initialize posterior as copy of prior
      posteriors.set(trait.name, this.priorToDistribution(prior, trait.type));
    }

    this.states.set(persona.id, {
      personaId: persona.id,
      priors,
      posteriors,
      updateCount: 0,
      lastUpdate: new Date(),
    });
  }

  /**
   * Update persona with new data points
   */
  updatePersona(
    persona: Persona,
    newDataPoints: DataPoint[],
    traitExtractor: (points: DataPoint[]) => Map<string, number | string[]>
  ): Persona {
    let state = this.states.get(persona.id);
    if (!state) {
      this.initializePersona(persona);
      state = this.states.get(persona.id)!;
    }

    // Extract trait values from new data
    const newTraitValues = traitExtractor(newDataPoints);

    // Update each trait using Bayesian inference
    const updatedTraits: Trait[] = [];

    for (const trait of persona.traits) {
      const newValue = newTraitValues.get(trait.name);
      if (newValue !== undefined) {
        const updatedTrait = this.updateTrait(trait, newValue, state);
        updatedTraits.push(updatedTrait);
      } else {
        updatedTraits.push(trait);
      }
    }

    // Update state
    state.updateCount++;
    state.lastUpdate = new Date();

    // Calculate new overall confidence
    const avgConfidence =
      updatedTraits.reduce((sum, t) => sum + t.confidence, 0) /
      updatedTraits.length;

    return {
      ...persona,
      traits: updatedTraits,
      overallConfidence: avgConfidence,
      version: persona.version + 1,
      lastUpdated: new Date(),
    };
  }

  /**
   * Update a single trait with new observation
   */
  private updateTrait(
    trait: Trait,
    newValue: number | string[],
    state: BayesianPersonaState
  ): Trait {
    const prior = state.priors.get(trait.name);
    if (!prior) {
      return trait;
    }

    // Handle different value types
    if (typeof newValue === 'number') {
      return this.updateNumericTrait(trait, newValue, prior, state);
    } else {
      return this.updateCategoricalTrait(trait, newValue, prior, state);
    }
  }

  /**
   * Bayesian update for numeric trait (using conjugate Beta-Binomial)
   */
  private updateNumericTrait(
    trait: Trait,
    newValue: number,
    prior: Prior,
    state: BayesianPersonaState
  ): Trait {
    // Apply decay to old observations
    const decayedAlpha = prior.alpha * this.config.decayFactor;
    const decayedBeta = prior.beta * this.config.decayFactor;

    // Normalize new value to [0, 1] for Beta distribution
    const normalizedValue = Math.max(0, Math.min(1, newValue));

    // Update as pseudo-observations
    const newAlpha = decayedAlpha + normalizedValue * this.config.learningRate;
    const newBeta = decayedBeta + (1 - normalizedValue) * this.config.learningRate;

    // Update prior
    prior.alpha = newAlpha;
    prior.beta = newBeta;
    prior.observations++;

    // Create updated distribution
    const posteriorDist = createBetaDistribution(
      newAlpha - 1,
      newBeta - 1,
      1,
      1
    );

    // Update state
    state.posteriors.set(trait.name, posteriorDist);

    // Calculate new confidence based on distribution spread
    const posteriorMean = distributionMean(posteriorDist);
    const posteriorVar = distributionVariance(posteriorDist);
    const confidence = Math.max(
      0,
      Math.min(1, 1 - 2 * Math.sqrt(posteriorVar))
    );

    return {
      ...trait,
      value: posteriorMean,
      confidence,
      sampleSize: prior.observations,
    };
  }

  /**
   * Bayesian update for categorical trait (using Dirichlet-Multinomial)
   */
  private updateCategoricalTrait(
    trait: Trait,
    newValues: string[],
    prior: Prior,
    state: BayesianPersonaState
  ): Trait {
    // Get existing categorical distribution
    let categories = new Map<string, number>();
    if (
      typeof trait.value === 'object' &&
      'categories' in (trait.value as Distribution)
    ) {
      categories = new Map((trait.value as Distribution).categories || []);
    }

    // Apply decay
    for (const [key, value] of categories) {
      categories.set(key, value * this.config.decayFactor);
    }

    // Add new observations
    for (const value of newValues) {
      const currentCount = categories.get(value) || 0;
      categories.set(value, currentCount + this.config.learningRate);
    }

    // Normalize to probabilities
    const total = Array.from(categories.values()).reduce((a, b) => a + b, 0);
    for (const [key, value] of categories) {
      categories.set(key, value / total);
    }

    // Update prior observation count
    prior.observations += newValues.length;

    // Create updated distribution
    const posteriorDist: Distribution = {
      type: 'categorical',
      categories,
    };

    state.posteriors.set(trait.name, posteriorDist);

    // Calculate confidence based on distribution entropy
    const entropy = this.calculateEntropy(categories);
    const maxEntropy = Math.log(categories.size);
    const confidence = maxEntropy > 0 ? 1 - entropy / maxEntropy : 1;

    return {
      ...trait,
      value: posteriorDist,
      confidence: Math.max(0, Math.min(1, confidence)),
      sampleSize: prior.observations,
    };
  }

  /**
   * Convert trait to prior distribution
   */
  private traitToPrior(trait: Trait): Prior {
    const strength = this.config.priorStrength;

    if (typeof trait.value === 'number') {
      // For numeric traits, use Beta prior
      const normalizedValue = Math.max(0, Math.min(1, trait.value));
      return {
        alpha: normalizedValue * strength + 1,
        beta: (1 - normalizedValue) * strength + 1,
        observations: trait.sampleSize || 0,
      };
    }

    // For categorical/distribution traits
    return {
      alpha: strength / 2,
      beta: strength / 2,
      observations: trait.sampleSize || 0,
    };
  }

  /**
   * Convert prior to distribution
   */
  private priorToDistribution(prior: Prior, traitType: string): Distribution {
    if (traitType === 'topic_frequency' || traitType === 'keyword_tfidf') {
      return {
        type: 'categorical',
        categories: new Map(),
      };
    }

    return {
      type: 'beta',
      alpha: prior.alpha,
      beta: prior.beta,
    };
  }

  /**
   * Calculate Shannon entropy of a categorical distribution
   */
  private calculateEntropy(categories: Map<string, number>): number {
    let entropy = 0;
    for (const p of categories.values()) {
      if (p > 0) {
        entropy -= p * Math.log(p);
      }
    }
    return entropy;
  }

  /**
   * Get posterior distribution for a trait
   */
  getPosterior(personaId: string, traitName: string): Distribution | undefined {
    const state = this.states.get(personaId);
    return state?.posteriors.get(traitName);
  }

  /**
   * Get credible interval for a trait (Bayesian confidence interval)
   */
  getCredibleInterval(
    personaId: string,
    traitName: string,
    credibility = 0.95
  ): [number, number] | undefined {
    const state = this.states.get(personaId);
    const prior = state?.priors.get(traitName);

    if (!prior) return undefined;

    // For Beta distribution, calculate credible interval
    const alpha = prior.alpha;
    const beta = prior.beta;

    // Use quantile function approximation
    const lower = this.betaQuantile(alpha, beta, (1 - credibility) / 2);
    const upper = this.betaQuantile(alpha, beta, (1 + credibility) / 2);

    return [lower, upper];
  }

  /**
   * Approximate quantile function for Beta distribution
   */
  private betaQuantile(alpha: number, beta: number, p: number): number {
    // Simple approximation using normal approximation for large alpha, beta
    const mean = alpha / (alpha + beta);
    const variance = (alpha * beta) / (Math.pow(alpha + beta, 2) * (alpha + beta + 1));
    const std = Math.sqrt(variance);

    // Normal quantile approximation
    const z = this.normalQuantileApprox(p);
    return Math.max(0, Math.min(1, mean + z * std));
  }

  private normalQuantileApprox(p: number): number {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;

    if (p > 0.5) {
      return -this.normalQuantileApprox(1 - p);
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
   * Get state for a persona
   */
  getState(personaId: string): BayesianPersonaState | undefined {
    return this.states.get(personaId);
  }

  /**
   * Reset persona state
   */
  resetPersona(personaId: string): void {
    this.states.delete(personaId);
  }

  /**
   * Export state for persistence
   */
  exportState(): Map<string, BayesianPersonaState> {
    return new Map(this.states);
  }

  /**
   * Import state from persistence
   */
  importState(states: Map<string, BayesianPersonaState>): void {
    this.states = new Map(states);
  }
}

/**
 * Convenience function for single update
 */
export function bayesianUpdatePersona(
  persona: Persona,
  newDataPoints: DataPoint[],
  traitExtractor: (points: DataPoint[]) => Map<string, number | string[]>,
  config?: Partial<BayesianConfig>
): Persona {
  const updater = new BayesianPersonaUpdater(config);
  return updater.updatePersona(persona, newDataPoints, traitExtractor);
}
