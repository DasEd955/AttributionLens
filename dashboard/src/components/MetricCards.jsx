/**
 * MetricCards.jsx - Four summary metric cards for the analytics dashboard.
 *
 * Displays the four principal metrics derived from the audit log:
 *
 *   1. Total submissions with verdict breakdown percentages.
 *   2. Appeal rate: appeals filed as a fraction of total decisions.
 *   3. Signal Disagreement Rate: mean |p_ai_llm - p_ai_style| and the
 *      percentage of cases where disagreement exceeded 0.40.
 *   4. Grounding Influence: mean absolute confidence delta from the
 *      grounding modifier, with boosted vs reduced breakdown.
 *
 * Props:
 *   stats (object): The JSON response from GET /dashboard/stats.
 *                   Renders nothing when null (loading or error state).
 */

import React from 'react'

/**
 * Format a decimal fraction as a percentage string.
 *
 * @param {number|null} val - Decimal value in [0, 1].
 * @param {number} [decimals=1] - Decimal places in the output.
 * @returns {string} Formatted percentage, e.g. "18.2%", or "--" when null.
 */
function pct(val, decimals = 1) {
  if (val == null) return '--'
  return (val * 100).toFixed(decimals) + '%'
}

/**
 * Format a float to a fixed number of decimal places.
 *
 * @param {number|null} val - The value to format.
 * @param {number} [decimals=2] - Decimal places in the output.
 * @returns {string} Formatted string, or "--" when null.
 */
function fmt(val, decimals = 2) {
  if (val == null) return '--'
  return Number(val).toFixed(decimals)
}

/**
 * Render four metric summary cards.
 *
 * Each card maps to one dashboard metric. Color coding matches the verdict
 * palette used in the charts: red for AI flagged content, green for human,
 * yellow for uncertain, blue/indigo for neutral aggregate metrics.
 *
 * @param {object} props
 * @param {object|null} props.stats - Aggregate stats from /dashboard/stats.
 * @returns {React.ReactElement|null} The card grid, or null when stats is falsy.
 */
export default function MetricCards({ stats }) {
  if (!stats) return null

  const {
    total,
    verdict_counts,
    appeal_rate,
    appeal_counts,
    signal_disagreement,
    grounding_influence,
  } = stats

  const vc = verdict_counts || {}
  const sd = signal_disagreement || {}
  const gi = grounding_influence || {}
  const ac = appeal_counts || {}

  const aiPct = total > 0 ? pct(vc.likely_ai / total) : '--'
  const humanPct = total > 0 ? pct(vc.likely_human / total) : '--'
  const uncertainPct = total > 0 ? pct(vc.uncertain / total) : '--'

  return (
    <div className="metric-grid">

      {/* Card 1: Total submissions + verdict split */}
      <div className="metric-card">
        <div className="label">Total Submissions</div>
        <div className="value" style={{ color: 'var(--accent)' }}>
          {total != null ? total.toLocaleString() : '--'}
        </div>
        <div className="sub">
          <span style={{ color: 'var(--ai)' }}>{aiPct} AI</span>
          {' / '}
          <span style={{ color: 'var(--uncertain)' }}>{uncertainPct} Uncertain</span>
          {' / '}
          <span style={{ color: 'var(--human)' }}>{humanPct} Human</span>
        </div>
      </div>

      {/* Card 2: Appeal rate */}
      <div className="metric-card">
        <div className="label">Appeal Rate</div>
        <div className="value" style={{ color: 'var(--uncertain)' }}>
          {pct(appeal_rate)}
        </div>
        <div className="sub">
          <span>{ac.total ?? '--'}</span> appeal{ac.total !== 1 ? 's' : ''} filed
          {' / '}
          <span>{ac.pending ?? '--'}</span> pending review
        </div>
      </div>

      {/* Card 3: Signal disagreement rate */}
      <div className="metric-card">
        <div className="label">Signal Disagreement Rate</div>
        <div className="value" style={{ color: 'var(--accent2)' }}>
          {fmt(sd.avg_disagreement)}
        </div>
        <div className="sub">
          avg |LLM - Stylometric|
          {' / '}
          <span>{pct(sd.pct_high_disagreement)}</span> above 0.40
        </div>
      </div>

      {/* Card 4: Grounding influence */}
      <div className="metric-card">
        <div className="label">Grounding Influence</div>
        <div className="value" style={{ color: 'var(--human)' }}>
          {fmt(gi.avg_influence)}
        </div>
        <div className="sub">
          avg |confidence delta|
          {' / '}
          <span style={{ color: 'var(--human)' }}>{pct(gi.pct_boosted)}</span> boosted
          {' / '}
          <span style={{ color: 'var(--ai)' }}>{pct(gi.pct_reduced)}</span> reduced
        </div>
      </div>

    </div>
  )
}
