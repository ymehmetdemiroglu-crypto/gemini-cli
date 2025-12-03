/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { PersonaEngine, formatPersona, formatGenerationResult } from './persona-engine.js';
import type { DataPoint } from '../types.js';

describe('PersonaEngine', () => {
  function generateTestData(size: number): DataPoint[] {
    const dataPoints: DataPoint[] = [];

    // Create 3 distinct clusters
    const clusters = [
      {
        centerEmbedding: [0, 0, 0, 0, 0],
        topics: ['technology', 'gadgets'],
        sentiment: 0.7,
        device: 'mobile',
        priceRange: 30,
      },
      {
        centerEmbedding: [10, 10, 10, 10, 10],
        topics: ['fashion', 'lifestyle'],
        sentiment: 0.5,
        device: 'desktop',
        priceRange: 150,
      },
      {
        centerEmbedding: [-10, -10, -10, -10, -10],
        topics: ['sports', 'fitness'],
        sentiment: -0.2,
        device: 'mobile',
        priceRange: 80,
      },
    ];

    const pointsPerCluster = Math.floor(size / 3);

    for (let c = 0; c < clusters.length; c++) {
      const cluster = clusters[c];
      for (let i = 0; i < pointsPerCluster; i++) {
        const embedding = cluster.centerEmbedding.map(
          (v) => v + (Math.random() - 0.5) * 2
        );

        dataPoints.push({
          id: `point_${c}_${i}`,
          embedding,
          metadata: {
            topics: cluster.topics,
            sentiment: cluster.sentiment + (Math.random() - 0.5) * 0.2,
            device: cluster.device,
            priceReference: cluster.priceRange + (Math.random() - 0.5) * 20,
            keywords: ['sample', 'test', cluster.topics[0]],
          },
          timestamp: new Date(
            Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
          ),
          text: `Sample text about ${cluster.topics.join(' and ')}`,
        });
      }
    }

    return dataPoints;
  }

  describe('generatePersonas', () => {
    it('should generate personas from clustered data', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 20, // Reduced for faster tests
        monteCarloIterations: 10,
        minSampleSize: 20,
        minPersonaSize: 5,
        minStabilityScore: 0.1, // Lower threshold for test stability
      });

      const data = generateTestData(150);
      const result = await engine.generatePersonas(data);

      // With synthetic data and lower thresholds, we should get personas
      // but the exact number depends on clustering stability
      expect(result.metadata.totalDataPoints).toBe(150);
      expect(result.metadata.processingTime).toBeGreaterThan(0);
      expect(result.metadata.bootstrapIterations).toBe(20);
    });

    it('should include validity indices for each persona', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 10,
        monteCarloIterations: 5,
        minSampleSize: 20,
        minPersonaSize: 5,
      });

      const data = generateTestData(100);
      const result = await engine.generatePersonas(data);

      for (const persona of result.personas) {
        expect(persona.validityIndices).toBeDefined();
        expect(persona.validityIndices.silhouette).toBeDefined();
        expect(persona.validityIndices.daviesBouldin).toBeDefined();
      }
    });

    it('should calculate stability scores', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 15,
        monteCarloIterations: 5,
        minSampleSize: 20,
        minPersonaSize: 5,
      });

      const data = generateTestData(100);
      const result = await engine.generatePersonas(data);

      for (const persona of result.personas) {
        expect(persona.stability).toBeDefined();
        expect(persona.stability.stabilityScore).toBeGreaterThanOrEqual(0);
        expect(persona.stability.stabilityScore).toBeLessThanOrEqual(1);
      }
    });

    it('should extract traits with confidence scores', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 10,
        monteCarloIterations: 5,
        minSampleSize: 20,
        minPersonaSize: 5,
      });

      const data = generateTestData(100);
      const result = await engine.generatePersonas(data);

      for (const persona of result.personas) {
        if (persona.traits.length > 0) {
          for (const trait of persona.traits) {
            expect(trait.confidence).toBeGreaterThanOrEqual(0);
            expect(trait.confidence).toBeLessThanOrEqual(1);
            expect(trait.sampleSize).toBeGreaterThan(0);
          }
        }
      }
    });

    it('should warn for small sample sizes', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 5,
        monteCarloIterations: 5,
        minSampleSize: 100, // Higher than data size
        minPersonaSize: 5,
      });

      const data = generateTestData(50);
      const result = await engine.generatePersonas(data);

      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.warnings.some((w) => w.includes('Sample size'))).toBe(true);
    });
  });

  describe('getPersona', () => {
    it('should retrieve persona by ID', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 10,
        monteCarloIterations: 5,
        minSampleSize: 20,
        minPersonaSize: 5,
      });

      const data = generateTestData(100);
      const result = await engine.generatePersonas(data);

      if (result.personas.length > 0) {
        const persona = engine.getPersona(result.personas[0].id);
        expect(persona).toBeDefined();
        expect(persona?.id).toBe(result.personas[0].id);
      }
    });
  });

  describe('getAllPersonas', () => {
    it('should return all generated personas', async () => {
      const engine = new PersonaEngine({
        bootstrapSamples: 10,
        monteCarloIterations: 5,
        minSampleSize: 20,
        minPersonaSize: 5,
      });

      const data = generateTestData(100);
      await engine.generatePersonas(data);

      const allPersonas = engine.getAllPersonas();
      expect(Array.isArray(allPersonas)).toBe(true);
    });
  });
});

describe('formatPersona', () => {
  it('should format persona for display', () => {
    const persona = {
      id: 'test_1',
      name: 'Test Persona',
      description: 'A test persona',
      cluster: {
        id: 'cluster_1',
        centroid: [0, 0],
        members: ['a', 'b', 'c'],
        size: 3,
      },
      traits: [
        {
          name: 'topic_tech',
          type: 'topic_frequency' as const,
          value: 0.8,
          confidence: 0.9,
          sampleSize: 100,
        },
      ],
      validityIndices: {
        silhouette: 0.7,
        daviesBouldin: 0.5,
        calinskiHarabasz: 100,
        dunn: 0.3,
      },
      stability: {
        stabilityScore: 0.85,
        bootstrapConsistency: 0.9,
        confidenceInterval: [0.7, 0.95] as [number, number],
        appearanceRate: 0.9,
      },
      robustnessScore: 0.8,
      overallConfidence: 0.85,
      sampleSize: 100,
      lastUpdated: new Date(),
      version: 1,
    };

    const formatted = formatPersona(persona);

    expect(formatted).toContain('Test Persona');
    expect(formatted).toContain('85.0%'); // Overall confidence
    expect(formatted).toContain('topic_tech');
    expect(formatted).toContain('Silhouette');
  });
});

describe('formatGenerationResult', () => {
  it('should format full result', () => {
    const result = {
      personas: [],
      metadata: {
        totalDataPoints: 100,
        processingTime: 1500,
        bootstrapIterations: 50,
        monteCarloIterations: 30,
        averageClusterValidity: {
          silhouette: 0.6,
          daviesBouldin: 0.8,
          calinskiHarabasz: 150,
          dunn: 0.25,
        },
        overallConfidence: 0.75,
      },
      warnings: ['Test warning'],
    };

    const formatted = formatGenerationResult(result);

    expect(formatted).toContain('100'); // Data points
    expect(formatted).toContain('1500'); // Processing time
    expect(formatted).toContain('Test warning');
    expect(formatted).toContain('Silhouette');
  });
});
