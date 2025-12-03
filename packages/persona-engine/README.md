# 🔬 Statistically-Validated Persona Generation Engine

A **scientifically-grounded** persona generation system that goes beyond AI summarization to create **data-backed, repeatable, and trustworthy** user personas.

## 🎯 What Makes This Different?

Traditional persona tools rely on LLM hallucinations and vibes. This engine uses:

- **Bootstrapped Sampling** - Personas are backed by statistical stability, not one-off clustering
- **Cluster Validity Indices** - Every cluster is validated with Silhouette, Davies-Bouldin, Calinski-Harabasz, and Dunn indices
- **Monte Carlo Validation** - Robustness scores from 100+ perturbation runs
- **Hypothesis Testing** - Every trait is statistically tested (t-tests, chi-square, ANOVA)
- **Bayesian Updating** - Incremental learning without full regeneration
- **Confidence Intervals** - Every measurement comes with uncertainty quantification

## 📊 The Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PERSONA GENERATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. EMBED DATA                                                       │
│     └─> Vector representation of user behavior                       │
│                                                                      │
│  2. BOOTSTRAPPED CLUSTERING                                          │
│     └─> 100+ samples, test cluster stability                         │
│     └─> If cluster appears in >50% of samples → statistically stable │
│                                                                      │
│  3. CLUSTER VALIDITY INDICES                                         │
│     ├─> Silhouette Score (-1 to 1, higher = better)                  │
│     ├─> Davies-Bouldin Index (lower = better)                        │
│     ├─> Calinski-Harabasz Index (higher = better)                    │
│     └─> Dunn Index (higher = better)                                 │
│                                                                      │
│  4. MONTE CARLO VALIDATION                                           │
│     └─> Perturb embeddings, recluster 100x                           │
│     └─> Robustness Score: 0.9 = strong, 0.3 = weak                   │
│                                                                      │
│  5. TRAIT EXTRACTION                                                 │
│     ├─> Topic frequencies with TF-IDF                                │
│     ├─> Sentiment distributions                                      │
│     ├─> Behavioral patterns                                          │
│     └─> Each trait tested with p < 0.05                              │
│                                                                      │
│  6. BAYESIAN UPDATES                                                 │
│     └─> New data → updated posteriors → gradual persona shifts       │
│                                                                      │
│  7. OUTPUT: PERSONAS WITH CONFIDENCE                                 │
│     └─> "Budget-conscious: 85% confidence (p=0.003)"                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```typescript
import { PersonaEngine, formatGenerationResult } from '@google/persona-engine';

// Initialize engine with configuration
const engine = new PersonaEngine({
  bootstrapSamples: 100,
  monteCarloIterations: 50,
  significanceLevel: 0.05,
  minConfidence: 0.5,
});

// Your data with embeddings and metadata
const dataPoints = [
  {
    id: 'user_1',
    embedding: [0.1, 0.5, 0.3, ...], // Pre-computed embedding
    metadata: {
      topics: ['technology', 'gadgets'],
      sentiment: 0.7,
      device: 'mobile',
      priceReference: 35,
      keywords: ['budget', 'value', 'specs'],
    },
    timestamp: new Date(),
  },
  // ... more data points
];

// Generate statistically-validated personas
const result = await engine.generatePersonas(dataPoints);

// Pretty print results
console.log(formatGenerationResult(result));

// Access individual personas
for (const persona of result.personas) {
  console.log(`${persona.name}: ${persona.overallConfidence * 100}% confidence`);
  console.log(`  Stability: ${persona.stability.stabilityScore}`);
  console.log(`  Robustness: ${persona.robustnessScore}`);
  
  for (const trait of persona.traits) {
    console.log(`  - ${trait.name}: ${trait.confidence * 100}% (p=${trait.pValue})`);
  }
}
```

## 📈 Example Output

```
═══════════════════════════════════════════════════════════════
📊 Tech-Savvy Budget Shopper
   ID: persona_1 | Version: 1
   Overall Confidence: 87.3%
───────────────────────────────────────────────────────────────
📝 A persona representing 142 data points. Primarily interested
   in technology and gadgets. Generally expresses positive sentiment.
   Most active during evening hours.
───────────────────────────────────────────────────────────────
📈 Statistical Validation:
   • Sample Size: 142
   • Stability Score: 89.2%
   • Robustness Score: 85.1% (strong)
   • Silhouette: 0.723
   • Davies-Bouldin: 0.412
───────────────────────────────────────────────────────────────
🏷️  Traits (with confidence):
   • topic_technology: 92% confidence (p=0.0012)
   • sentiment_positive: 78% confidence (p=0.0234)
   • price_budget_conscious: 85% confidence (p=0.0089)
   • posting_time_evening: 71% confidence (p=0.0456)
   • device_mobile: 88% confidence (p=0.0023)
═══════════════════════════════════════════════════════════════
```

## 🧪 Bayesian Updates

Instead of regenerating personas from scratch, update them incrementally:

```typescript
// New data arrives
const newDataPoints = [...];

// Update personas with Bayesian inference
const updatedPersonas = engine.updateWithNewData(newDataPoints);

// Personas shift gradually, no random resets
console.log(`Version: ${updatedPersonas[0].version}`);
```

## 📦 Package Structure

```
persona-engine/
├── src/
│   ├── statistics/          # Distributions, hypothesis testing
│   │   ├── distributions.ts  # Normal, Beta, Chi-square, etc.
│   │   └── hypothesis-testing.ts  # t-tests, ANOVA, chi-square
│   │
│   ├── clustering/          # Clustering algorithms
│   │   ├── kmeans.ts        # K-Means with k-means++ init
│   │   ├── hdbscan.ts       # Density-based clustering
│   │   └── bootstrap.ts     # Bootstrapped sampling
│   │
│   ├── validation/          # Statistical validation
│   │   ├── cluster-validity.ts  # CVI metrics
│   │   └── monte-carlo.ts   # Robustness testing
│   │
│   ├── bayesian/            # Incremental updates
│   │   └── bayesian-updater.ts  # Bayesian persona updating
│   │
│   ├── traits/              # Trait extraction
│   │   └── trait-extractor.ts  # Statistical trait analysis
│   │
│   ├── engine/              # Main orchestrator
│   │   └── persona-engine.ts
│   │
│   └── types.ts             # TypeScript interfaces
│
├── examples/
│   └── demo.ts              # Full working demo
│
└── README.md
```

## ⚙️ Configuration Options

```typescript
interface PersonaEngineConfig {
  // Clustering
  clusteringAlgorithm: 'kmeans' | 'hdbscan';
  minClusterSize: number;      // Default: 10
  maxClusters: number;         // Default: 10

  // Bootstrap sampling
  bootstrapSamples: number;    // Default: 100
  sampleSize: number;          // Default: 0.8 (80% per sample)
  minSampleSize: number;       // Default: 50

  // Monte Carlo validation
  monteCarloIterations: number;    // Default: 50
  perturbationMagnitude: number;   // Default: 0.1

  // Statistical thresholds
  significanceLevel: number;   // Default: 0.05
  minConfidence: number;       // Default: 0.5
  minStabilityScore: number;   // Default: 0.4

  // Bayesian updating
  enableBayesianUpdates: boolean;  // Default: true
  priorStrength: number;       // Default: 10

  // Output
  minPersonaSize: number;      // Default: 10
  maxTraitsPerPersona: number; // Default: 15
}
```

## 🔢 Statistical Metrics Explained

### Cluster Validity Indices

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Silhouette** | -1 to 1 | Higher = better cluster separation |
| **Davies-Bouldin** | 0+ | Lower = more distinct clusters |
| **Calinski-Harabasz** | 0+ | Higher = denser, well-separated |
| **Dunn** | 0+ | Higher = better ratio of inter/intra cluster distance |

### Confidence Interpretation

| Score | Rating | Meaning |
|-------|--------|---------|
| 0.9+ | Strong | Highly robust, trustworthy |
| 0.7-0.9 | Moderate | Reliable with some uncertainty |
| 0.5-0.7 | Weak | Use with caution |
| <0.5 | Unstable | Not recommended for decisions |

## 🏃 Running the Demo

```bash
cd packages/persona-engine
npx tsx examples/demo.ts
```

## 🧪 Running Tests

```bash
cd packages/persona-engine
npm test
```

## 📄 License

Apache 2.0
