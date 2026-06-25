/**
 * SignalHeatmap.jsx - Scatterplot of LLM score vs stylometric score, colored by verdict.
 *
 * Each dot is one submission. The X-axis is the stylometric signal probability
 * (p_ai_style) and the Y-axis is the LLM signal probability (p_ai_llm). Dot
 * color encodes the verdict: red for likely_ai, green for likely_human, and
 * yellow for uncertain.
 *
 * Agreement clusters (dots near the diagonal) and disagreement spread (dots
 * far from it) are immediately visible. The uncertain band, populated when
 * signals disagree, directly demonstrates the confidence-collapse mechanism
 * that protects human writers from false AI accusations.
 *
 * Props:
 *   points (Array): Array of scatter point objects from GET /dashboard/scatter.
 *     Each entry is { content_id, p_ai_llm, p_ai_style, verdict }.
 *   loading (boolean): When true, renders a loading placeholder.
 *   error (string|null): When non-null, renders the error message.
 */

import React from 'react'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

/** Dot color by verdict, matching the CSS palette. */
const VERDICT_COLOR = {
  likely_ai: '#f87171',
  likely_human: '#34d399',
  uncertain: '#fbbf24',
}

/** Human-readable label for each verdict key. */
const VERDICT_LABEL = {
  likely_ai: 'Likely AI',
  likely_human: 'Likely Human',
  uncertain: 'Uncertain',
}

/**
 * Custom dot renderer that uses the verdict color from each data point.
 *
 * Recharts Scatter does not natively color individual dots by a data field,
 * so this renderer reads the verdict from the payload and picks the color.
 *
 * @param {object} props - Recharts dot props injected by ScatterChart.
 * @returns {React.ReactElement} An SVG circle element.
 */
function VerdictDot(props) {
  const { cx, cy, payload } = props
  const color = VERDICT_COLOR[payload.verdict] || '#8892a4'
  return <circle cx={cx} cy={cy} r={4} fill={color} fillOpacity={0.75} stroke="none" />
}

/**
 * Custom tooltip for the scatterplot.
 *
 * Shows both signal scores and the verdict for the hovered dot.
 *
 * @param {object} props - Recharts tooltip props (active, payload).
 * @returns {React.ReactElement|null} Tooltip element, or null when inactive.
 */
function CustomTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const d = payload[0]?.payload
  if (!d) return null
  const color = VERDICT_COLOR[d.verdict] || 'var(--text-muted)'
  return (
    <div style={{
      background: 'var(--surface2)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
    }}>
      <div style={{ fontWeight: 600, color, marginBottom: 6 }}>
        {VERDICT_LABEL[d.verdict] || d.verdict}
      </div>
      <div style={{ color: 'var(--text-muted)' }}>
        LLM score: <span style={{ color: 'var(--text)' }}>{Number(d.p_ai_llm).toFixed(3)}</span>
      </div>
      <div style={{ color: 'var(--text-muted)' }}>
        Stylometric: <span style={{ color: 'var(--text)' }}>{Number(d.p_ai_style).toFixed(3)}</span>
      </div>
    </div>
  )
}

/**
 * Partition a flat array of points into three arrays by verdict.
 *
 * Recharts Scatter requires one <Scatter> element per color series, so the
 * points must be split before rendering.
 *
 * @param {Array} points - Raw scatter point objects.
 * @returns {{ likely_ai: Array, likely_human: Array, uncertain: Array }}
 */
function partitionByVerdict(points) {
  const groups = { likely_ai: [], likely_human: [], uncertain: [] }
  for (const p of points) {
    const key = p.verdict in groups ? p.verdict : 'uncertain'
    groups[key].push(p)
  }
  return groups
}

/**
 * Render the LLM vs stylometric signal scatterplot.
 *
 * A diagonal reference line (x = y) is drawn to make signal agreement
 * visually obvious: dots on the line agree perfectly; dots far from it
 * are the high disagreement cases that route into the uncertain verdict.
 *
 * @param {object} props
 * @param {Array} props.points - Scatter point objects from /dashboard/scatter.
 * @param {boolean} props.loading - Whether data is still being fetched.
 * @param {string|null} props.error - Fetch error message, if any.
 * @returns {React.ReactElement} The scatterplot or a state placeholder.
 */
export default function SignalHeatmap({ points, loading, error }) {
  if (loading) return <div className="state-box">Loading...</div>
  if (error) return <div className="state-box error-box">{error}</div>
  if (!points || points.length === 0) {
    return <div className="state-box">No signal data yet -- submit some text first.</div>
  }

  const groups = partitionByVerdict(points)

  // Legend items rendered manually because Recharts legend with custom dots
  // does not reliably pick up colors from the shape prop.
  const legendItems = Object.entries(VERDICT_LABEL).map(([key, label]) => (
    <span key={key} style={{ marginRight: 16, fontSize: 11, color: 'var(--text-muted)' }}>
      <svg width="10" height="10" style={{ marginRight: 4, verticalAlign: 'middle' }}>
        <circle cx="5" cy="5" r="4" fill={VERDICT_COLOR[key]} fillOpacity={0.85} />
      </svg>
      {label}
    </span>
  ))

  return (
    <div>
      <div style={{ marginBottom: 12 }}>{legendItems}</div>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="p_ai_style"
            domain={[0, 1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'Stylometric', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="p_ai_llm"
            domain={[0, 1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'LLM', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--text-muted)', fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          {/* Agreement diagonal: dots on this line have zero disagreement */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="var(--border)"
            strokeDasharray="5 5"
          />
          {Object.entries(groups).map(([verdict, data]) => (
            <Scatter
              key={verdict}
              name={VERDICT_LABEL[verdict]}
              data={data}
              shape={<VerdictDot />}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
