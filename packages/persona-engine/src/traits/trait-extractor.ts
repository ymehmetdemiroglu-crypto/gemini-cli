/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Trait extraction from data points with statistical confidence
 */

import type { Trait, DataPoint, HypothesisTestResult } from '../types.js';
import {
  createDistributionFromData,
  createCategoricalDistribution,
  createHistogramDistribution,
} from '../statistics/distributions.js';
import {
  oneSampleTTest,
  chiSquareGoodnessOfFit,
  confidenceIntervalMean,
} from '../statistics/hypothesis-testing.js';

/**
 * Configuration for trait extraction
 */
export interface TraitExtractionConfig {
  minSampleSize: number;
  significanceLevel: number;
  minConfidence: number;
  maxTraitsPerType: number;
}

const DEFAULT_EXTRACTION_CONFIG: TraitExtractionConfig = {
  minSampleSize: 10,
  significanceLevel: 0.05,
  minConfidence: 0.5,
  maxTraitsPerType: 10,
};

/**
 * Extract traits from data points with statistical validation
 */
export class TraitExtractor {
  private config: TraitExtractionConfig;

  constructor(config: Partial<TraitExtractionConfig> = {}) {
    this.config = { ...DEFAULT_EXTRACTION_CONFIG, ...config };
  }

  /**
   * Extract all traits from a cluster of data points
   */
  extractTraits(
    dataPoints: DataPoint[],
    globalStats?: {
      topicFrequencies: Map<string, number>;
      avgSentiment: number;
      avgPostingHour: number;
    }
  ): Trait[] {
    const traits: Trait[] = [];

    if (dataPoints.length < this.config.minSampleSize) {
      return traits;
    }

    // Extract different types of traits
    traits.push(...this.extractTopicTraits(dataPoints, globalStats?.topicFrequencies));
    traits.push(...this.extractSentimentTraits(dataPoints, globalStats?.avgSentiment));
    traits.push(...this.extractTemporalTraits(dataPoints, globalStats?.avgPostingHour));
    traits.push(...this.extractBehaviorTraits(dataPoints));
    traits.push(...this.extractKeywordTraits(dataPoints));

    // Filter by confidence and significance
    return traits.filter(
      (t) => t.confidence >= this.config.minConfidence
    );
  }

  /**
   * Extract topic frequency traits
   */
  private extractTopicTraits(
    dataPoints: DataPoint[],
    globalTopics?: Map<string, number>
  ): Trait[] {
    const traits: Trait[] = [];
    const topicCounts = new Map<string, number>();

    // Count topics in this cluster
    for (const point of dataPoints) {
      const topics = point.metadata?.topics as string[] | undefined;
      if (topics) {
        for (const topic of topics) {
          topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
        }
      }
    }

    // Calculate distribution
    const totalTopicMentions = Array.from(topicCounts.values()).reduce(
      (a, b) => a + b,
      0
    );
    if (totalTopicMentions === 0) return traits;

    // Convert to frequencies
    const topicFreqs = new Map<string, number>();
    for (const [topic, count] of topicCounts) {
      topicFreqs.set(topic, count / totalTopicMentions);
    }

    // Test significance against global distribution
    const sortedTopics = Array.from(topicCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.config.maxTraitsPerType);

    for (const [topic, count] of sortedTopics) {
      const clusterFreq = count / totalTopicMentions;
      const globalFreq = globalTopics?.get(topic) || 0.1;

      // Calculate confidence based on deviation from global
      const deviation = Math.abs(clusterFreq - globalFreq) / Math.max(globalFreq, 0.01);
      const confidence = Math.min(1, deviation / 2 + 0.5);

      // Statistical test
      const testResult = this.testProportionDifference(
        count,
        totalTopicMentions,
        globalFreq
      );

      if (testResult.isSignificant || confidence > this.config.minConfidence) {
        traits.push({
          name: `topic_${topic}`,
          type: 'topic_frequency',
          value: clusterFreq,
          confidence: testResult.isSignificant ? Math.max(confidence, 0.8) : confidence,
          pValue: testResult.pValue,
          sampleSize: count,
          testType: 'chi_square',
        });
      }
    }

    return traits;
  }

  /**
   * Extract sentiment traits
   */
  private extractSentimentTraits(
    dataPoints: DataPoint[],
    globalAvgSentiment?: number
  ): Trait[] {
    const traits: Trait[] = [];

    const sentiments: number[] = [];
    for (const point of dataPoints) {
      const sentiment = point.metadata?.sentiment as number | undefined;
      if (sentiment !== undefined) {
        sentiments.push(sentiment);
      }
    }

    if (sentiments.length < this.config.minSampleSize) return traits;

    const distribution = createDistributionFromData(sentiments);
    const mean = distribution.mean || 0;

    // Test against global or neutral
    const baseline = globalAvgSentiment ?? 0;
    const testResult = oneSampleTTest(
      sentiments,
      baseline,
      this.config.significanceLevel
    );

    // Interpret sentiment
    let label: string;
    if (mean > 0.3) label = 'positive';
    else if (mean < -0.3) label = 'negative';
    else label = 'neutral';

    traits.push({
      name: `sentiment_${label}`,
      type: 'sentiment',
      value: distribution,
      confidence: testResult.isSignificant
        ? Math.min(1, 0.7 + Math.abs(testResult.effectSize || 0) * 0.3)
        : 0.5,
      pValue: testResult.pValue,
      sampleSize: sentiments.length,
      testType: 't_test',
    });

    return traits;
  }

  /**
   * Extract temporal/posting time traits
   */
  private extractTemporalTraits(
    dataPoints: DataPoint[],
    globalAvgHour?: number
  ): Trait[] {
    const traits: Trait[] = [];

    const hours: number[] = [];
    const daysOfWeek: string[] = [];

    for (const point of dataPoints) {
      if (point.timestamp) {
        const date = new Date(point.timestamp);
        hours.push(date.getHours());
        daysOfWeek.push(
          ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][date.getDay()]
        );
      }
    }

    if (hours.length < this.config.minSampleSize) return traits;

    // Hour distribution
    const hourDist = createHistogramDistribution(hours, 24);
    const avgHour = hours.reduce((a, b) => a + b, 0) / hours.length;

    // Determine time category
    let timeCategory: string;
    if (avgHour >= 6 && avgHour < 12) timeCategory = 'morning';
    else if (avgHour >= 12 && avgHour < 17) timeCategory = 'afternoon';
    else if (avgHour >= 17 && avgHour < 21) timeCategory = 'evening';
    else timeCategory = 'night';

    // Test significance
    const baseline = globalAvgHour ?? 12;
    const testResult = oneSampleTTest(
      hours,
      baseline,
      this.config.significanceLevel
    );

    traits.push({
      name: `posting_time_${timeCategory}`,
      type: 'posting_time',
      value: hourDist,
      confidence: testResult.isSignificant ? 0.85 : 0.6,
      pValue: testResult.pValue,
      sampleSize: hours.length,
      testType: 't_test',
    });

    // Day of week distribution
    const weekendCount = daysOfWeek.filter(
      (d) => d === 'Sat' || d === 'Sun'
    ).length;
    const weekendRatio = weekendCount / daysOfWeek.length;

    if (weekendRatio > 0.4) {
      traits.push({
        name: 'weekend_active',
        type: 'behavior',
        value: weekendRatio,
        confidence: 0.7,
        sampleSize: daysOfWeek.length,
      });
    } else if (weekendRatio < 0.2) {
      traits.push({
        name: 'weekday_active',
        type: 'behavior',
        value: 1 - weekendRatio,
        confidence: 0.7,
        sampleSize: daysOfWeek.length,
      });
    }

    return traits;
  }

  /**
   * Extract behavioral traits
   */
  private extractBehaviorTraits(dataPoints: DataPoint[]): Trait[] {
    const traits: Trait[] = [];

    // Device usage
    const devices: string[] = [];
    for (const point of dataPoints) {
      const device = point.metadata?.device as string | undefined;
      if (device) devices.push(device);
    }

    if (devices.length >= this.config.minSampleSize) {
      const deviceDist = createCategoricalDistribution(devices);
      const categories = deviceDist.categories || new Map();

      // Find dominant device
      let maxDevice = '';
      let maxFreq = 0;
      for (const [device, freq] of categories) {
        if (freq > maxFreq) {
          maxFreq = freq;
          maxDevice = device;
        }
      }

      if (maxFreq > 0.5) {
        traits.push({
          name: `device_${maxDevice}`,
          type: 'device_usage',
          value: deviceDist,
          confidence: Math.min(1, maxFreq + 0.2),
          sampleSize: devices.length,
        });
      }
    }

    // Price sensitivity
    const prices: number[] = [];
    for (const point of dataPoints) {
      const priceRef = point.metadata?.priceReference as number | undefined;
      if (priceRef !== undefined) prices.push(priceRef);
    }

    if (prices.length >= this.config.minSampleSize) {
      const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
      const distribution = createDistributionFromData(prices);

      let sensitivity: string;
      if (avgPrice < 50) sensitivity = 'budget_conscious';
      else if (avgPrice < 200) sensitivity = 'moderate_spender';
      else sensitivity = 'premium_buyer';

      const ci = confidenceIntervalMean(prices, 0.95);
      const intervalWidth = ci[1] - ci[0];
      const confidence = Math.max(0.5, 1 - intervalWidth / (avgPrice * 2));

      traits.push({
        name: `price_${sensitivity}`,
        type: 'price_sensitivity',
        value: distribution,
        confidence,
        sampleSize: prices.length,
      });
    }

    return traits;
  }

  /**
   * Extract keyword/TF-IDF traits
   */
  private extractKeywordTraits(dataPoints: DataPoint[]): Trait[] {
    const traits: Trait[] = [];

    // Build term frequency
    const termFreq = new Map<string, number>();
    let totalTerms = 0;

    for (const point of dataPoints) {
      const keywords = point.metadata?.keywords as string[] | undefined;
      if (keywords) {
        for (const keyword of keywords) {
          termFreq.set(keyword, (termFreq.get(keyword) || 0) + 1);
          totalTerms++;
        }
      }
    }

    if (totalTerms === 0) return traits;

    // Calculate TF-IDF scores (simplified - using TF * log(N/df))
    const N = dataPoints.length;
    const docFreq = new Map<string, number>();

    for (const point of dataPoints) {
      const keywords = new Set(point.metadata?.keywords as string[] | undefined);
      for (const keyword of keywords) {
        docFreq.set(keyword, (docFreq.get(keyword) || 0) + 1);
      }
    }

    const tfidfScores: Array<{ term: string; score: number; count: number }> = [];

    for (const [term, tf] of termFreq) {
      const df = docFreq.get(term) || 1;
      const idf = Math.log(N / df);
      const tfidf = (tf / totalTerms) * idf;
      tfidfScores.push({ term, score: tfidf, count: tf });
    }

    // Sort by TF-IDF and take top keywords
    tfidfScores.sort((a, b) => b.score - a.score);
    const topKeywords = tfidfScores.slice(0, this.config.maxTraitsPerType);

    // Normalize scores for confidence
    const maxScore = topKeywords[0]?.score || 1;

    for (const { term, score, count } of topKeywords) {
      const confidence = Math.min(1, (score / maxScore) * 0.7 + 0.3);

      traits.push({
        name: `keyword_${term}`,
        type: 'keyword_tfidf',
        value: score,
        confidence,
        sampleSize: count,
      });
    }

    return traits;
  }

  /**
   * Test if a proportion differs significantly from expected
   */
  private testProportionDifference(
    successes: number,
    total: number,
    expectedProportion: number
  ): HypothesisTestResult {
    const expected = total * expectedProportion;
    const expectedFailures = total * (1 - expectedProportion);

    return chiSquareGoodnessOfFit(
      [successes, total - successes],
      [expected, expectedFailures],
      this.config.significanceLevel
    );
  }
}

/**
 * Calculate overall confidence for a set of traits
 */
export function calculateOverallTraitConfidence(traits: Trait[]): number {
  if (traits.length === 0) return 0;

  // Weighted average based on sample size
  let weightedSum = 0;
  let totalWeight = 0;

  for (const trait of traits) {
    const weight = Math.log(trait.sampleSize + 1);
    weightedSum += trait.confidence * weight;
    totalWeight += weight;
  }

  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

/**
 * Format trait for display
 */
export function formatTrait(trait: Trait): string {
  const confidencePercent = Math.round(trait.confidence * 100);
  const value =
    typeof trait.value === 'number'
      ? `${Math.round(trait.value * 100)}%`
      : trait.value;

  return `${trait.name}: ${value} (${confidencePercent}% confidence, n=${trait.sampleSize})`;
}
