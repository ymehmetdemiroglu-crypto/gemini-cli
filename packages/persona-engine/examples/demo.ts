/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * 🔬 Statistically-Validated Persona Generation Engine Demo
 *
 * This demo shows the full pipeline:
 * 1. Generate synthetic user data with distinct behavioral patterns
 * 2. Run bootstrapped clustering for stability
 * 3. Validate clusters with statistical indices
 * 4. Monte Carlo validation for robustness
 * 5. Extract traits with hypothesis testing
 * 6. Output scientifically-grounded personas
 *
 * Run with: npx tsx examples/demo.ts
 */

import {
  PersonaEngine,
  formatGenerationResult,
  type DataPoint,
} from '../src/index.js';

// ============================================================================
// 🎲 Data Generation - Create realistic synthetic user data
// ============================================================================

function generateSyntheticUserData(totalUsers: number): DataPoint[] {
  const dataPoints: DataPoint[] = [];

  // Define distinct user segments with statistical properties
  const segments = [
    {
      name: 'Tech-Savvy Budget Shoppers',
      center: [0.8, 0.2, 0.9, 0.3, 0.7],
      topics: ['technology', 'gadgets', 'deals', 'reviews'],
      sentimentMean: 0.6,
      sentimentStd: 0.15,
      device: 'mobile',
      priceRangeMean: 35,
      priceRangeStd: 15,
      peakHour: 20, // Evening
      keywords: ['budget', 'value', 'specs', 'comparison', 'affordable'],
    },
    {
      name: 'Premium Lifestyle Enthusiasts',
      center: [0.2, 0.9, 0.3, 0.85, 0.4],
      topics: ['fashion', 'luxury', 'travel', 'wellness'],
      sentimentMean: 0.75,
      sentimentStd: 0.1,
      device: 'desktop',
      priceRangeMean: 250,
      priceRangeStd: 80,
      peakHour: 14, // Afternoon
      keywords: ['premium', 'exclusive', 'quality', 'luxury', 'designer'],
    },
    {
      name: 'Fitness & Health Focused',
      center: [0.5, 0.4, 0.6, 0.5, 0.9],
      topics: ['fitness', 'nutrition', 'sports', 'health'],
      sentimentMean: 0.4,
      sentimentStd: 0.25,
      device: 'mobile',
      priceRangeMean: 75,
      priceRangeStd: 30,
      peakHour: 7, // Morning
      keywords: ['workout', 'protein', 'results', 'progress', 'goals'],
    },
    {
      name: 'Content Creators & Influencers',
      center: [0.7, 0.6, 0.4, 0.7, 0.5],
      topics: ['social-media', 'content', 'trends', 'engagement'],
      sentimentMean: 0.55,
      sentimentStd: 0.2,
      device: 'both',
      priceRangeMean: 120,
      priceRangeStd: 50,
      peakHour: 16, // Late afternoon
      keywords: ['viral', 'followers', 'aesthetic', 'collab', 'brand'],
    },
  ];

  const usersPerSegment = Math.floor(totalUsers / segments.length);

  for (let segIdx = 0; segIdx < segments.length; segIdx++) {
    const segment = segments[segIdx];

    for (let i = 0; i < usersPerSegment; i++) {
      // Generate embedding with noise around segment center
      const embedding = segment.center.map((c) => {
        const noise = gaussianRandom(0, 0.15);
        return Math.max(0, Math.min(1, c + noise));
      });

      // Generate realistic metadata
      const sentiment =
        segment.sentimentMean + gaussianRandom(0, segment.sentimentStd);
      const priceRef =
        segment.priceRangeMean + gaussianRandom(0, segment.priceRangeStd);

      // Generate posting time with peak hour preference
      const hour = Math.round(
        segment.peakHour + gaussianRandom(0, 3)
      ) % 24;
      const timestamp = new Date();
      timestamp.setHours(hour < 0 ? hour + 24 : hour);
      timestamp.setDate(timestamp.getDate() - Math.floor(Math.random() * 60));

      // Select random subset of keywords
      const selectedKeywords = segment.keywords
        .filter(() => Math.random() > 0.4)
        .slice(0, 3 + Math.floor(Math.random() * 3));

      // Randomly select device based on segment tendency
      let device = segment.device;
      if (device === 'both') {
        device = Math.random() > 0.5 ? 'mobile' : 'desktop';
      }
      // Add some noise to device selection
      if (Math.random() < 0.1) {
        device = device === 'mobile' ? 'desktop' : 'mobile';
      }

      dataPoints.push({
        id: `user_${segIdx}_${i}`,
        embedding,
        text: `Sample content about ${segment.topics.slice(0, 2).join(' and ')}`,
        timestamp,
        source: 'synthetic',
        metadata: {
          topics: segment.topics.filter(() => Math.random() > 0.3),
          sentiment: Math.max(-1, Math.min(1, sentiment)),
          device,
          priceReference: Math.max(5, priceRef),
          keywords: selectedKeywords,
          engagementScore: Math.random() * 100,
          sessionDuration: 30 + Math.random() * 300,
          originalSegment: segment.name, // For validation purposes
        },
      });
    }
  }

  // Shuffle the data
  return dataPoints.sort(() => Math.random() - 0.5);
}

/**
 * Box-Muller transform for Gaussian random numbers
 */
function gaussianRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z * std + mean;
}

// ============================================================================
// 🚀 Main Demo
// ============================================================================

async function runDemo() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🔬 STATISTICALLY-VALIDATED PERSONA GENERATION ENGINE                       ║
║                                                                              ║
║   This demo generates scientifically-grounded personas using:                ║
║   • Bootstrapped sampling for cluster stability                              ║
║   • Cluster validity indices (Silhouette, DB, CH, Dunn)                      ║
║   • Monte Carlo validation for robustness testing                            ║
║   • Bayesian updating for incremental learning                               ║
║   • Hypothesis testing for trait validation                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);

  // Step 1: Generate synthetic data
  console.log('📊 Step 1: Generating synthetic user data...');
  const totalUsers = 500;
  const dataPoints = generateSyntheticUserData(totalUsers);
  console.log(`   ✅ Generated ${dataPoints.length} data points\n`);

  // Step 2: Initialize the persona engine
  console.log('⚙️  Step 2: Initializing Persona Engine...');
  const engine = new PersonaEngine({
    // Clustering configuration
    clusteringAlgorithm: 'kmeans',
    minClusterSize: 10,
    maxClusters: 6,

    // Bootstrap sampling for stability
    bootstrapSamples: 30,
    sampleSize: 0.8,
    minSampleSize: 30,

    // Monte Carlo for robustness
    monteCarloIterations: 20,
    perturbationMagnitude: 0.05,

    // Statistical thresholds - lower for demo
    significanceLevel: 0.05,
    minConfidence: 0.3,
    minStabilityScore: 0.1, // Lower threshold to find clusters

    // Bayesian updating
    enableBayesianUpdates: true,
    priorStrength: 10,

    // Output configuration
    minPersonaSize: 10,
    maxTraitsPerPersona: 12,
  });
  console.log('   ✅ Engine initialized\n');

  // Step 3: Generate personas
  console.log('🔄 Step 3: Running the full persona generation pipeline...');
  console.log('   - Bootstrapped clustering for stability analysis');
  console.log('   - Computing cluster validity indices');
  console.log('   - Running Monte Carlo validation');
  console.log('   - Extracting traits with hypothesis testing');
  console.log('');

  const startTime = Date.now();
  const result = await engine.generatePersonas(dataPoints);
  const totalTime = Date.now() - startTime;

  console.log(`   ✅ Pipeline completed in ${totalTime}ms\n`);

  // Step 4: Display results
  console.log(formatGenerationResult(result));

  // Step 5: Show comparison with ground truth
  console.log('\n📈 VALIDATION AGAINST GROUND TRUTH:');
  console.log('════════════════════════════════════════════════════════════');

  const segmentMapping = new Map<string, Set<string>>();
  for (const point of dataPoints) {
    const segment = point.metadata?.originalSegment as string;
    if (!segmentMapping.has(segment)) {
      segmentMapping.set(segment, new Set());
    }
    segmentMapping.get(segment)!.add(point.id);
  }

  for (const persona of result.personas) {
    console.log(`\n🎭 ${persona.name} (Cluster ID: ${persona.id})`);

    // Check overlap with ground truth segments
    const memberSet = new Set(persona.cluster.members);
    console.log('   Ground truth segment overlap:');

    for (const [segment, members] of segmentMapping) {
      const overlap = [...memberSet].filter((id) => members.has(id)).length;
      const percentage = (overlap / persona.sampleSize * 100).toFixed(1);
      if (overlap > 0) {
        console.log(`   • ${segment}: ${overlap} (${percentage}%)`);
      }
    }
  }

  // Step 6: Demonstrate Bayesian updating
  console.log('\n\n🔄 BAYESIAN UPDATE DEMONSTRATION:');
  console.log('════════════════════════════════════════════════════════════');
  console.log('   Simulating new data arrival...\n');

  // Generate some new data points
  const newData = generateSyntheticUserData(50);
  console.log(`   Generated ${newData.length} new data points`);

  // Update personas
  const updatedPersonas = engine.updateWithNewData(newData);
  console.log(`   Updated ${updatedPersonas.length} personas with Bayesian inference`);

  if (updatedPersonas.length > 0) {
    const updated = updatedPersonas[0];
    console.log(`\n   Example: ${updated.name}`);
    console.log(`   • Version: ${updated.version}`);
    console.log(`   • Updated confidence: ${(updated.overallConfidence * 100).toFixed(1)}%`);
  }

  // Final summary
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🎉 DEMO COMPLETE                                                           ║
║                                                                              ║
║   Key Statistics:                                                            ║
║   • Total data points processed: ${result.metadata.totalDataPoints.toString().padStart(4)}                                   ║
║   • Personas generated: ${result.personas.length}                                                   ║
║   • Average confidence: ${(result.metadata.overallConfidence * 100).toFixed(1)}%                                          ║
║   • Processing time: ${result.metadata.processingTime}ms                                            ║
║                                                                              ║
║   Each persona includes:                                                     ║
║   ✓ Bootstrap stability scores                                               ║
║   ✓ Cluster validity indices (Silhouette, DB, CH, Dunn)                      ║
║   ✓ Monte Carlo robustness scores                                            ║
║   ✓ Statistically-validated traits with p-values                             ║
║   ✓ Confidence intervals for all measurements                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);
}

// Run the demo
runDemo().catch(console.error);
