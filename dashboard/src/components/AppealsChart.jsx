/**
 * AppealsChart.jsx - Bar chart showing appeal volume and status breakdown.
 *
 * Renders a simple grouped bar chart with three bars derived from the
 * appeal_counts block of GET /dashboard/stats:
 *
 *   - Filed: total appeals submitted.
 *   - Pending: decisions currently in under_review status.
 *   - Resolved: filed minus pending (proxy for closed reviews).
 *
 * Because the audit log does not yet record explicit "upheld/rejected"
 * outcomes, "Resolved" is derived as filed - pending. This is noted in the
 * chart subtitle so the viewer is not misled.
 *
 * Props:
 *   stats (object): The JSON response from GET /dashboard/stats. Only the
 *     appeal_counts and appeal_rate fields are used.
 *   loading (boolean): When true, renders a loading placeholder.
 *   error (string|null): When non-null, renders the error message.
 */

import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from 'recharts'

/** Bar colors for each appeal status category. */
const BAR_COLORS = {
  Filed: '#818cf8',
  Pending: '#fbbf24',
  Resolved: '#34d399',
}

/**
 * Custom tooltip for the appeals bar chart.
 *
 * Shows the category name and count on hover.
 *
 * @param {object} props - Recharts tooltip props (active, payload, label).
 * @returns {React.ReactElement|null} Tooltip element, or null when inactive.
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div style={{
      background: 'var(--surface2)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
    }}>
      <div style={{ fontWeight: 600, color: payload[0]?.fill, marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--text)' }}>
        Count: <strong>{payload[0]?.value ?? '--'}</strong>
      </div>
    </div>
  )
}

/**
 * Render the appeals status bar chart.
 *
 * Derives the three bar dataset from the stats prop and wraps the chart in a
 * ResponsiveContainer so it fills its parent card without a fixed pixel width.
 *
 * @param {object} props
 * @param {object|null} props.stats - Aggregate stats from /dashboard/stats.
 * @param {boolean} props.loading - Whether data is still being fetched.
 * @param {string|null} props.error - Fetch error message, if any.
 * @returns {React.ReactElement} The chart or a state placeholder.
 */
export default function AppealsChart({ stats, loading, error }) {
  if (loading) return <div className="state-box">Loading...</div>
  if (error) return <div className="state-box error-box">{error}</div>
  if (!stats) return null

  const ac = stats.appeal_counts || {}
  const filed = ac.total ?? 0
  const pending = ac.pending ?? 0
  const resolved = Math.max(0, filed - pending)

  const data = [
    { name: 'Filed', value: filed },
    { name: 'Pending', value: pending },
    { name: 'Resolved', value: resolved },
  ]

  if (filed === 0) {
    return <div className="state-box">No appeals filed yet.</div>
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
        <YAxis allowDecimals={false} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={72}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={BAR_COLORS[entry.name]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
