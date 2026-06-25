/**
 * VerdictBarChart.jsx - Stacked bar chart of daily verdict distribution.
 *
 * Renders a Recharts BarChart where each bar represents one calendar day and
 * is stacked into three segments: Likely Human (green), Uncertain (yellow),
 * and Likely AI (red). This visualizes how detection patterns change over
 * time and makes the uncertain zone, the intended safe harbour for
 * disagreeing signals, immediately visible.
 *
 * Props:
 *   timeseries (Array): Array of daily entries from GET /dashboard/timeseries.
 *     Each entry is { date, likely_ai, likely_human, uncertain }.
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
  Legend,
  ResponsiveContainer,
} from 'recharts'

/** Palette that matches the CSS variables in index.css. */
const COLORS = {
  likely_human: '#34d399',
  uncertain: '#fbbf24',
  likely_ai: '#f87171',
}

/**
 * Custom tooltip shown on bar hover.
 *
 * Displays the full date and a count for each verdict category so the
 * viewer can read exact numbers without relying on bar height alone.
 *
 * @param {object} props - Recharts tooltip props (active, payload, label).
 * @returns {React.ReactElement|null} Tooltip element, or null when inactive.
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null
  const total = payload.reduce((sum, p) => sum + (p.value || 0), 0)
  return (
    <div style={{
      background: 'var(--surface2)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text)' }}>{label}</div>
      {[...payload].reverse().map((p) => (
        <div key={p.dataKey} style={{ color: p.fill, marginBottom: 2 }}>
          {p.name}: <strong>{p.value}</strong>
        </div>
      ))}
      <div style={{ color: 'var(--text-muted)', marginTop: 6, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
        Total: <strong style={{ color: 'var(--text)' }}>{total}</strong>
      </div>
    </div>
  )
}

/**
 * Shorten a YYYY-MM-DD date string to MM/DD for the axis label.
 *
 * @param {string} dateStr - ISO date string.
 * @returns {string} Short label like "06/24".
 */
function shortDate(dateStr) {
  if (!dateStr) return ''
  const parts = dateStr.split('-')
  if (parts.length < 3) return dateStr
  return `${parts[1]}/${parts[2]}`
}

/**
 * Render the stacked verdict distribution bar chart.
 *
 * When the timeseries array is empty an explanatory placeholder is shown
 * rather than an empty chart, so the UI degrades gracefully on a fresh
 * installation with no submissions yet.
 *
 * @param {object} props
 * @param {Array} props.timeseries - Daily verdict counts.
 * @param {boolean} props.loading - Whether data is still being fetched.
 * @param {string|null} props.error - Fetch error message, if any.
 * @returns {React.ReactElement} The chart or a state placeholder.
 */
export default function VerdictBarChart({ timeseries, loading, error }) {
  if (loading) return <div className="state-box">Loading...</div>
  if (error) return <div className="state-box error-box">{error}</div>
  if (!timeseries || timeseries.length === 0) {
    return <div className="state-box">No submissions recorded yet.</div>
  }

  const data = timeseries.map((d) => ({ ...d, date: shortDate(d.date) }))

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis allowDecimals={false} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Legend
          iconType="circle"
          iconSize={8}
          formatter={(value) => (
            <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
              {value === 'likely_human' ? 'Likely Human'
                : value === 'likely_ai' ? 'Likely AI'
                : 'Uncertain'}
            </span>
          )}
        />
        <Bar dataKey="likely_human" name="likely_human" stackId="a" fill={COLORS.likely_human} radius={[0, 0, 0, 0]} />
        <Bar dataKey="uncertain" name="uncertain" stackId="a" fill={COLORS.uncertain} />
        <Bar dataKey="likely_ai" name="likely_ai" stackId="a" fill={COLORS.likely_ai} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
