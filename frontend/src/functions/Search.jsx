import React, { useState } from 'react'

// Backend API base URL for constructing absolute asset URLs
const API_BASE = import.meta.env.VITE_API_BASE

export function SearchVideoResult({ video }) {
    const [selectedImage, setSelectedImage] = useState(null)
    const keyframes = video.keyframes || []
    const detectedLabels = video.detected_objects || []

    // Ensure keyframe URLs are absolute by prepending API_BASE if needed
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

    return (
        <div className="search-result-card video-result">
            <div className="result-header">
                <h3>📹 {video.filename}</h3>
                <span className="result-score">Match: {video.score}</span>
            </div>
            {video.snippet && (
                <div className="result-snippet">
                    <strong>Context:</strong> ...{video.snippet}...
                </div>
            )}
            {detectedLabels.length > 0 && (
                <div className="detected-labels">
                    {detectedLabels.map((obj, idx) => {
                        const label = obj.label || obj.name || 'unknown'
                        const conf = obj.confidence || obj.conf || 0
                        return (
                            <span key={idx} className="label-badge">
                                {label} {conf > 0 && <small>({(conf * 100).toFixed(0)}%)</small>}
                            </span>
                        )
                    })}
                </div>
            )}
            {keyframes.length > 0 && (
                <div className="result-keyframes">
                    <strong>Keyframes:</strong>
                    <div className="keyframes-list">
                        {keyframesWithAbsoluteUrls.slice(0, 6).map((kf, i) => {
                            const ts = kf.timestamp
                            const tsDisplay = ts != null && typeof ts === 'number' && ts.toFixed ? `${ts.toFixed(2)}s` : (ts != null ? `${ts}s` : '—')
                            return (
                                <div key={i} className="result-keyframe-row">
                                    <img
                                        src={kf.url}
                                        alt={`frame-${i}`}
                                        onClick={() => setSelectedImage(kf.url)}
                                        className="result-keyframe-img"
                                    />
                                    <div className="result-keyframe-col">
                                        <div className="result-keyframe-time">{tsDisplay}</div>
                                        {kf.filename && <div className="result-keyframe-filename">{kf.filename}</div>}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
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

export function SearchAudioResult({ audio, onSimilarText }) {
    return (
        <div className="search-result-card audio-result">
            <div className="result-header">
                <h3>🎵 {audio.filename}</h3>
                <span className="result-score">Match: {audio.score}</span>
            </div>
            {audio.snippet && (
                <div className="result-snippet">
                    <strong>Context:</strong> ...{audio.snippet}...
                </div>
            )}
            <div className="result-transcription">
                <strong>Transcription:</strong>
                <p className="transcription-text">{audio.transcriptions}</p>
                    {onSimilarText && (
                    <button
                        className="search-btn mt-8"
                        onClick={() => onSimilarText(audio.transcriptions)}
                    >
                        🔍 Find Similar
                    </button>
                )}
            </div>
        </div>
    )
}

export function SearchSimilarityResult({ result, onSimilarImage, onSimilarText }) {
    const [selectedImage, setSelectedImage] = useState(null)

    if (result.type === 'video') {
        const imageUrl = result.metadata?.url
        const absoluteUrl = imageUrl && !imageUrl.startsWith('http') ? `${API_BASE}${imageUrl}` : imageUrl

        return (
            <div className="search-result-card similarity-result video-result">
                <div className="result-header">
                    <h3>📹 {result.filename}</h3>
                    <span className="result-score">Similarity: {(result.score * 100).toFixed(1)}%</span>
                </div>
                {absoluteUrl && (
                    <div className="result-keyframes">
                        <img
                            src={absoluteUrl}
                            alt="match"
                            onClick={() => setSelectedImage(absoluteUrl)}
                            className="similarity-image"
                        />
                    </div>
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

    if (result.type === 'audio') {
        return (
            <div className="search-result-card similarity-result audio-result">
                <div className="result-header">
                    <h3>🎵 {result.filename}</h3>
                    <span className="result-score">Similarity: {(result.score * 100).toFixed(1)}%</span>
                </div>
                <div className="result-transcription">
                    <strong>Text:</strong>
                    <p className="transcription-text">{result.text}</p>
                    {onSimilarText && (
                        <button
                            className="search-btn mt-8"
                            onClick={() => onSimilarText(result.text)}
                        >
                            🔍 Find More Similar
                        </button>
                    )}
                </div>
            </div>
        )
    }

    return null
}

export function SearchSection({ isSearching, onSearch }) {
    const [searchQuery, setSearchQuery] = useState('')

    // Invoke parent search handler when user submits
    const handleSearch = async () => {
        if (searchQuery.trim().length > 0) {
            onSearch(searchQuery, 1)
        }
    }

    const handleClear = () => {
        setSearchQuery('')
        onSearch('')
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSearch()
        }
    }

    return (
        <section className="search-section">
            <h2>🔍 Search Media</h2>
            <div className="search-container">
                <div className="search-field">
                    <label>Search Videos & Audios:</label>
                    <div className="search-input-group">
                        <input
                            type="text"
                            placeholder="Search by filename, transcription, or detected objects..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyPress={handleKeyPress}
                            disabled={isSearching}
                        />
                        <button
                            onClick={handleSearch}
                            disabled={isSearching || !searchQuery.trim()}
                            className="search-btn"
                        >
                            {isSearching ? 'Searching...' : 'Search'}
                        </button>
                        {searchQuery && <button onClick={handleClear} className="clear-btn">Clear</button>}
                    </div>
                </div>
            </div>
        </section>
    )
}