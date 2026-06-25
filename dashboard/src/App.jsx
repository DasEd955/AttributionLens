/**
 * App.jsx - Root component and data-fetching orchestrator for the AttributionLens
 * analytics dashboard.
 *
 * Fetches three endpoints from the Flask backend on mount and on each manual
 * refresh, then passes the results down to the four visualization components:
 *
 *   GET /api/dashboard/stats      -> MetricCards, AppealsChart
 *   GET /api/dashboard/timeseries -> VerdictBarChart
 *   GET /api/dashboard/scatter    -> SignalHeatmap
 *
 * The Vite proxy (vite.config.js) rewrites /api/* to http://localhost:5000/*,
 * so all three fetches work in development without any CORS configuration on
 * the client side.
 *
 * Layout:
 *   Header (title + last-updated timestamp + refresh button)
 *   MetricCards row (4 summary cards)
 *   Chart grid (2x2):
 *     [VerdictBarChart]  [AppealsChart]
 *     [SignalHeatmap spanning full width]
 */

import React, { useCallback, useEffect, useState } from 'react'
import MetricCards from './components/MetricCards.jsx'
import VerdictBarChart from './components/VerdictBarChart.jsx'
import AppealsChart from './components/AppealsChart.jsx'
import SignalHeatmap from './components/SignalHeatmap.jsx'

/** Base URL prefix; the Vite proxy maps /api -> http://localhost:5000. */
const API = '/api'

/**
 * Fetch JSON from a URL, returning { data, error }.
 *
 * Never throws: network errors and non-OK HTTP responses are both captured
 * into the error field so callers can handle them uniformly.
 *
 * @param {string} url - The URL to fetch.
 * @returns {Promise<{ data: any|null, error: string|null }>}
 */
async function fetchJson(url) {
  try {
    const res = await fetch(url)
    if (!res.ok) return { data: null, error: `HTTP ${res.status}` }
    const data = await res.json()
    return { data, error: null }
  } catch (e) {
    return { data: null, error: e.message || 'Network error' }
  }
}

/**
 * Format a Date as a human-readable "Last updated" string.
 *
 * @param {Date|null} date - The date to format, or null before the first fetch.
 * @returns {string} Formatted string like "Last updated: 2:34:07 PM".
 */
function formatUpdated(date) {
  if (!date) return ''
  return 'Last updated: ' + date.toLocaleTimeString()
}

/**
 * Root dashboard component.
 *
 * Manages three independent fetch states (stats, timeseries, scatter) so
 * each chart section can show its own loading/error state without blocking
 * the others. All three fetches are fired in parallel on mount and on
 * each refresh.
 *
 * @returns {React.ReactElement} The full dashboard page.
 */
export default function App() {
  const [stats, setStats] = useState(null)
  const [statsError, setStatsError] = useState(null)

  const [timeseries, setTimeseries] = useState(null)
  const [timeseriesError, setTimeseriesError] = useState(null)

  const [points, setPoints] = useState(null)
  const [scatterError, setScatterError] = useState(null)

  const [loading, setLoading] = useState(true)
  const [updatedAt, setUpdatedAt] = useState(null)

  /**
   * Fire all three data fetches in parallel and update state.
   *
   * Marked as useCallback so the effect dependency array stays stable and
   * the refresh button can call the same function without an extra closure.
   */
  const fetchAll = useCallback(async () => {
    setLoading(true)

    const [statsRes, timeRes, scatterRes] = await Promise.all([
      fetchJson(`${API}/dashboard/stats`),
      fetchJson(`${API}/dashboard/timeseries?days=30`),
      fetchJson(`${API}/dashboard/scatter?limit=500`),
    ])

    setStats(statsRes.data)
    setStatsError(statsRes.error)

    setTimeseries(timeRes.data?.timeseries ?? null)
    setTimeseriesError(timeRes.error)

    setPoints(scatterRes.data?.points ?? null)
    setScatterError(scatterRes.error)

    setLoading(false)
    setUpdatedAt(new Date())
  }, [])

  // Fetch on first render.
  useEffect(() => { fetchAll() }, [fetchAll])

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="dashboard-header-row">
          <div>
            <h1>AttributionLens Analytics</h1>
            <p>
              Detection patterns, appeal rates, signal disagreement, and grounding influence
              {updatedAt && (
                <span style={{ marginLeft: 12, color: 'var(--border)' }}>|</span>
              )}
              {updatedAt && (
                <span style={{ marginLeft: 12 }}>{formatUpdated(updatedAt)}</span>
              )}
            </p>
          </div>
          <button className="refresh-btn" onClick={fetchAll} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Four metric summary cards */}
      {statsError ? (
        <div className="state-box error-box" style={{ marginBottom: 24 }}>
          Failed to load stats: {statsError}
        </div>
      ) : (
        <MetricCards stats={stats} />
      )}

      {/* 2x2 chart grid */}
      <div className="chart-grid">

        {/* Chart 1: Detection patterns over time */}
        <div className="chart-card">
          <h2>Detection Patterns</h2>
          <p className="chart-subtitle">
            Verdict distribution per day (last 30 days)
          </p>
          <VerdictBarChart
            timeseries={timeseries}
            loading={loading && timeseries === null}
            error={timeseriesError}
          />
        </div>

        {/* Chart 2: Appeal breakdown */}
        <div className="chart-card">
          <h2>Appeal Status</h2>
          <p className="chart-subtitle">
            Filed / Pending / Resolved (resolved = filed - pending)
          </p>
          <AppealsChart
            stats={stats}
            loading={loading && stats === null}
            error={statsError}
          />
        </div>

        {/* Chart 3: Signal agreement scatterplot -- full width */}
        <div className="chart-card" style={{ gridColumn: '1 / -1' }}>
          <h2>Signal Agreement Heatmap</h2>
          <p className="chart-subtitle">
            Each dot is one submission. X = stylometric score, Y = LLM score, color = verdict.
            Dots near the diagonal agree; dots far from it drove the uncertain verdict.
          </p>
          <SignalHeatmap
            points={points}
            loading={loading && points === null}
            error={scatterError}
          />
        </div>

      </div>
    </div>
  )
}
