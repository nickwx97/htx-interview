import React, { useState } from 'react'

export default function AudioCard({ audio, onSimilarText }) {
    // If audio has `rows`, show each transcription separately
    if (audio.rows && Array.isArray(audio.rows)) {
        return (
            <div className="card audio-card">
                <div className="card-title">{audio.filename} {audio.id ? `(id: ${audio.id})` : ''}</div>
                <div className="audio-rows">
                    {audio.rows.map(r => {
                        // Parse timestamps to get start and end times (handles object or array formats)
                        let timeRange = 'N/A'
                        if (r.timestamps) {
                            try {
                                let ts = r.timestamps
                                // Parse string to object if needed
                                if (typeof ts === 'string') {
                                    ts = JSON.parse(ts)
                                }
                                if (ts && typeof ts === 'object') {
                                    // Handle both {start, end} and [start, end] formats
                                    const start = ts.start !== undefined && ts.start !== null ? parseFloat(ts.start) : (ts[0] !== undefined && ts[0] !== null ? parseFloat(ts[0]) : undefined)
                                    const end = ts.end !== undefined && ts.end !== null ? parseFloat(ts.end) : (ts[1] !== undefined && ts[1] !== null ? parseFloat(ts[1]) : undefined)
                                    if (start !== undefined && end !== undefined && !isNaN(start) && !isNaN(end)) {
                                        timeRange = `${start.toFixed(2)}s - ${end.toFixed(2)}s`
                                    }
                                }
                            } catch (e) { console.error('Timestamp parse error:', e)/* ignore */ }
                        }

                        // extract confidence if present in timestamps or directly on the row
                        let confidenceVal = null
                        try {
                            let ts = r.timestamps
                            if (typeof ts === 'string') ts = JSON.parse(ts)
                            if (ts && typeof ts === 'object') {
                                if (ts.confidence !== undefined && ts.confidence !== null) confidenceVal = parseFloat(ts.confidence)
                                else if (ts.conf !== undefined && ts.conf !== null) confidenceVal = parseFloat(ts.conf)
                            }
                            if (confidenceVal == null && r.confidence !== undefined) confidenceVal = parseFloat(r.confidence)
                        } catch (e) { confidenceVal = null }

                        return (
                            <div key={r.id} className="audio-row">
                                <div className="row-meta row-meta-strong">{timeRange}</div>
                                <div className="row-transcription">
                                    {r.transcriptions}
                                    {confidenceVal != null && !isNaN(confidenceVal) && (
                                        <span className="confidence-badge"> {(confidenceVal * 100).toFixed(0)}% </span>
                                    )}
                                    <button className="row-search-btn ml-8" title="Find similar" onClick={() => onSimilarText && onSimilarText(r.transcriptions)}>🔍</button>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        )
    }

    return (
        <div className="card audio-card">
            <div className="card-title">{audio.filename} {audio.id ? `(id: ${audio.id})` : ''}</div>
            <div className="transcription">{audio.transcriptions}</div>
        </div>
    )
}