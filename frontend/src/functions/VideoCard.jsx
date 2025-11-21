import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE

export default function VideoCard({ video, onSimilarImage }) {
    const keyframes = video.keyframes || []
    const [selectedImage, setSelectedImage] = useState(null)
    const [currentKeyframePage, setCurrentKeyframePage] = useState(1)
    const KEYFRAMES_PER_PAGE = 6

    // Check and prepare keyframe URLs
    const keyframesWithAbsoluteUrls = keyframes.map(kf => {
        const url = kf.url.startsWith('http') ? kf.url : `${API_BASE}${kf.url}`
        // try to parse frame_idx from filename if provided by backend
        let frame_idx = null
        try {
            if (kf.filename) {
                const name = kf.filename.replace(/\.[^/.]+$/, "")
                if (name.startsWith('kf_')) frame_idx = parseInt(name.split('kf_').pop())
            }
        } catch (e) { frame_idx = null }
        return { ...kf, url, frame_idx }
    })

    // Build a map of keyframe index -> detected objects by grouping DB rows
    const objectsByKeyframeIdx = {}
    if (video.rows && Array.isArray(video.rows) && video.rows.length > 0) {
        const groups = []
        let currentGroup = null
        for (let ri = 0; ri < video.rows.length; ri++) {
            const row = video.rows[ri]
            if (!row) continue
            // prefer explicit frame_idx from DB row when present
            let rowFrameIdx = null
            if (row.frame_idx !== undefined && row.frame_idx !== null) {
                rowFrameIdx = Number(row.frame_idx)
            }
            let tsVal = null
            try {
                tsVal = row.frame_timestamps ? JSON.parse(row.frame_timestamps) : null
            } catch (e) {
                tsVal = row.frame_timestamps
            }

            // start new group if timestamp changes
            if (!currentGroup || currentGroup.timestamp !== tsVal) {
                currentGroup = { timestamp: tsVal, objects: [], frame_idx: rowFrameIdx }
                groups.push(currentGroup)
            }

            // if current group doesn't have a frame_idx yet, set it from row
            if (currentGroup && currentGroup.frame_idx == null && rowFrameIdx != null) {
                currentGroup.frame_idx = rowFrameIdx
            }

            // parse detected_objects (may be a JSON string of an object or array)
            try {
                const parsed = row.detected_objects ? JSON.parse(row.detected_objects) : null
                if (parsed) {
                    if (Array.isArray(parsed)) {
                        currentGroup.objects.push(...parsed)
                    } else {
                        currentGroup.objects.push(parsed)
                    }
                }
            } catch (e) {
                // if parsing fails, skip
            }
        }

        // Map grouped frames to keyframe indices. Prefer explicit `frame_idx` when available
        const leftovers = []
        for (let gi = 0; gi < groups.length; gi++) {
            const g = groups[gi]
            if (g.frame_idx != null) {
                // find position of this frame_idx in the keyframes array
                const pos = keyframesWithAbsoluteUrls.findIndex(k => k.frame_idx === Number(g.frame_idx))
                if (pos >= 0) {
                    objectsByKeyframeIdx[pos] = g
                    continue
                }
            }
            leftovers.push(g)
        }

        // Fill remaining keyframe positions with leftover groups in order
        if (leftovers.length > 0) {
            for (let k = 0; k < keyframesWithAbsoluteUrls.length && leftovers.length > 0; k++) {
                if (!objectsByKeyframeIdx.hasOwnProperty(k)) {
                    objectsByKeyframeIdx[k] = leftovers.shift()
                }
            }
        }
    }

    // Get the first row's ID for similarity search
    const dbId = video.rows && video.rows.length > 0 ? video.rows[0].id : null

    // Paginate keyframes
    const totalKeyframePages = Math.ceil(keyframesWithAbsoluteUrls.length / KEYFRAMES_PER_PAGE)
    const keyframeStartIdx = (currentKeyframePage - 1) * KEYFRAMES_PER_PAGE
    const keyframeEndIdx = keyframeStartIdx + KEYFRAMES_PER_PAGE
    const paginatedKeyframes = keyframesWithAbsoluteUrls.slice(keyframeStartIdx, keyframeEndIdx)

    return (
        <div className="card video-card">
            <div className="card-title">{video.filename} {video.id ? `(id: ${video.id})` : ''}</div>
            {keyframes.length > 0 && (
                <>
                    <div className="keyframes-info">
                        Showing keyframes {keyframeStartIdx + 1}-{Math.min(keyframeEndIdx, keyframes.length)} of {keyframes.length}
                    </div>
                    <div className="keyframes-container">
                        {paginatedKeyframes.map((kf, i) => {
                            // Calculate actual keyframe index in full array for object lookup
                            const actualKeyframeIdx = keyframeStartIdx + i
                            const frameObjects = objectsByKeyframeIdx[actualKeyframeIdx] || { timestamp: kf.timestamp, objects: [] }
                            const objs = frameObjects.objects || []

                            return (
                                <div key={actualKeyframeIdx} className="keyframe-wrapper">
                                    <div className="keyframe-card">
                                        <div className="kf-img-wrap">
                                            <img
                                                src={kf.url}
                                                alt={`keyframe-${actualKeyframeIdx}`}
                                                className="keyframe-image"
                                                onClick={() => setSelectedImage(kf.url)}
                                            />
                                            <button className="kf-search-btn" title="Find similar" onClick={() => onSimilarImage && onSimilarImage(dbId)}>🔍</button>
                                        </div>
                                        <div className="keyframe-details">
                                            <div className="keyframe-timestamp">
                                                {(() => {
                                                    const dt = frameObjects && frameObjects.timestamp != null ? frameObjects.timestamp : kf.timestamp
                                                    if (typeof dt === 'number' && !isNaN(dt)) return `${dt.toFixed(2)}s`
                                                    if (dt != null) return `${dt}s`
                                                    return '—'
                                                })()}
                                            </div>
                                            {objs.length > 0 ? (
                                                <div className="keyframe-objects">
                                                    <strong className="detected-title">Detected Objects:</strong>
                                                    {objs.map((obj, idx) => {
                                                        const label = obj.label || obj.name || 'unknown'
                                                        const conf = obj.confidence || obj.conf || 0
                                                        return (
                                                            <div key={idx} className="object-row">
                                                                <span className="object-label">{label}</span>
                                                                {conf > 0 && <span className="object-conf">({(conf * 100).toFixed(0)}%)</span>}
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            ) : (
                                                <div className="no-objects">No objects detected</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>

                    {totalKeyframePages > 1 && (
                        <div className="keyframe-pagination">
                            <button
                                onClick={() => setCurrentKeyframePage(Math.max(1, currentKeyframePage - 1))}
                                disabled={currentKeyframePage === 1}
                                className="pagination-btn"
                            >
                                ← Previous
                            </button>

                            <div className="pagination-numbers">
                                {Array.from({ length: totalKeyframePages }, (_, i) => i + 1).map(pageNum => (
                                    <button
                                        key={pageNum}
                                        onClick={() => setCurrentKeyframePage(pageNum)}
                                        className={`page-number ${currentKeyframePage === pageNum ? 'active' : ''}`}
                                    >
                                        {pageNum}
                                    </button>
                                ))}
                            </div>

                            <button
                                onClick={() => setCurrentKeyframePage(Math.min(totalKeyframePages, currentKeyframePage + 1))}
                                disabled={currentKeyframePage === totalKeyframePages}
                                className="pagination-btn"
                            >
                                Next →
                            </button>
                        </div>
                    )}
                </>
            )}

            {selectedImage && (
                <div className="image-modal" onClick={() => setSelectedImage(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <span className="modal-close" onClick={() => setSelectedImage(null)}>&times;</span>
                        <img src={selectedImage} alt="enlarged" className="modal-image" />
                    </div>
                </div>
            )}
        </div>
    )
}