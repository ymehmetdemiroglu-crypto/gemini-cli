/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Simple demo showing the persona engine components working
 */

import {
  KMeans,
  buildClusters,
  calculateClusterValidityIndices,
  TraitExtractor,
  formatTrait,
  type DataPoint,
} from '../src/index.js';

// Generate well-separated synthetic data
function generateData(): DataPoint[] {
  const dataPoints: DataPoint[] = [];

  // Cluster 1: Tech enthusiasts (centered at [1, 0, 0])
  for (let i = 0; i < 50; i++) {
    dataPoints.push({
      id: `tech_${i}`,
      embedding: [
        1 + (Math.random() - 0.5) * 0.3,
        0 + (Math.random() - 0.5) * 0.3,
        0 + (Math.random() - 0.5) * 0.3,
      ],
      metadata: {
        topics: ['technology', 'gadgets', 'software'],
        sentiment: 0.6 + Math.random() * 0.3,
        device: Math.random() > 0.3 ? 'mobile' : 'desktop',
        priceReference: 30 + Math.random() * 40,
        keywords: ['tech', 'innovation', 'digital'],
      },
      timestamp: new Date(),
    });
  }

  // Cluster 2: Fashion lovers (centered at [0, 1, 0])
  for (let i = 0; i < 50; i++) {
    dataPoints.push({
      id: `fashion_${i}`,
      embedding: [
        0 + (Math.random() - 0.5) * 0.3,
        1 + (Math.random() - 0.5) * 0.3,
        0 + (Math.random() - 0.5) * 0.3,
      ],
      metadata: {
        topics: ['fashion', 'lifestyle', 'luxury'],
        sentiment: 0.7 + Math.random() * 0.2,
        device: Math.random() > 0.7 ? 'mobile' : 'desktop',
        priceReference: 150 + Math.random() * 100,
        keywords: ['style', 'trendy', 'premium'],
      },
      timestamp: new Date(),
    });
  }

  // Cluster 3: Fitness focused (centered at [0, 0, 1])
  for (let i = 0; i < 50; i++) {
    dataPoints.push({
      id: `fitness_${i}`,
      embedding: [
        0 + (Math.random() - 0.5) * 0.3,
        0 + (Math.random() - 0.5) * 0.3,
        1 + (Math.random() - 0.5) * 0.3,
      ],
      metadata: {
        topics: ['fitness', 'health', 'sports'],
        sentiment: 0.4 + Math.random() * 0.3,
        device: 'mobile',
        priceReference: 60 + Math.random() * 40,
        keywords: ['workout', 'fitness', 'healthy'],
      },
      timestamp: new Date(),
    });
  }

  return dataPoints;
}

async function main() {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║  🔬 PERSONA ENGINE - SIMPLE DEMO                           ║
╚════════════════════════════════════════════════════════════╝
`);

  // Step 1: Generate data
  console.log('📊 Step 1: Generating synthetic data...');
  const dataPoints = generateData();
  console.log(`   ✅ Generated ${dataPoints.length} data points in 3 clusters\n`);

  // Step 2: Run K-Means clustering
  console.log('🎯 Step 2: Running K-Means clustering...');
  const embeddings = dataPoints.map((d) => d.embedding);
  const kmeans = new KMeans(3);
  const assignments = kmeans.fit(embeddings);
  const clusters = buildClusters(dataPoints, assignments, kmeans.getCentroids());
  console.log(`   ✅ Found ${clusters.length} clusters\n`);

  // Step 3: Calculate cluster validity
  console.log('📈 Step 3: Calculating Cluster Validity Indices...');
  const validity = calculateClusterValidityIndices(dataPoints, clusters);
  console.log(`   • Silhouette Score: ${validity.silhouette.toFixed(3)} (higher is better, max 1.0)`);
  console.log(`   • Davies-Bouldin Index: ${validity.daviesBouldin.toFixed(3)} (lower is better)`);
  console.log(`   • Calinski-Harabasz: ${validity.calinskiHarabasz.toFixed(1)} (higher is better)`);
  console.log(`   • Dunn Index: ${validity.dunn.toFixed(3)} (higher is better)\n`);

  // Step 4: Extract traits for each cluster
  console.log('🏷️  Step 4: Extracting traits with statistical validation...\n');
  const traitExtractor = new TraitExtractor({
    minSampleSize: 5,
    significanceLevel: 0.05,
    minConfidence: 0.3,
  });

  for (let i = 0; i < clusters.length; i++) {
    const cluster = clusters[i];
    const clusterPoints = dataPoints.filter((d) =>
      cluster.members.includes(d.id)
    );

    console.log(`═══════════════════════════════════════════════════════════`);
    console.log(`📊 CLUSTER ${i + 1}: ${cluster.size} members`);
    console.log(`───────────────────────────────────────────────────────────`);

    // Extract traits
    const traits = traitExtractor.extractTraits(clusterPoints);

    if (traits.length > 0) {
      console.log(`🏷️  Extracted Traits:`);
      for (const trait of traits.slice(0, 8)) {
        console.log(`   ${formatTrait(trait)}`);
      }
    } else {
      console.log(`   No significant traits found`);
    }

    // Show sample member IDs
    console.log(`\n👥 Sample members: ${cluster.members.slice(0, 5).join(', ')}...`);
    console.log('');
  }

  console.log(`═══════════════════════════════════════════════════════════`);
  console.log(`
✅ DEMO COMPLETE!

The persona engine successfully:
  • Clustered 150 data points into 3 distinct groups
  • Calculated statistical validity indices
  • Extracted traits with confidence scores

Each trait includes:
  • Confidence percentage (based on statistical tests)
  • Sample size
  • p-value (when hypothesis testing was applied)
`);
}

main().catch(console.error);
