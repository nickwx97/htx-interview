import axios from 'axios';
import React, { useState, useEffect } from 'react';
import AudioCard from './functions/AudioCard';
import PaginationControls from './functions/Pagination';
import { SearchSimilarityResult, SearchSection, SearchVideoResult, SearchAudioResult } from './functions/Search';
import UploadForm from './functions/UploadForm';
import VideoCard from './functions/VideoCard';

const API_BASE = import.meta.env.VITE_API_BASE

export default function App() {
    const [videos, setVideos] = useState([]);
    const [audios, setAudios] = useState([]);
    const [searchResults, setSearchResults] = useState([]);
    const [searchPagination, setSearchPagination] = useState(null);
    const [currentSearchPage, setCurrentSearchPage] = useState(1);
    const [isSearching, setIsSearching] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [activeVideoTab, setActiveVideoTab] = useState(0);
    const [activeAudioTab, setActiveAudioTab] = useState(0);
    const [similarityModal, setSimilarityModal] = useState({ open: false, type: null, query: null, matches: [] });

    // Pagination for videos and audios lists (10 items per page)
    const [currentVideoPage, setCurrentVideoPage] = useState(1);
    const [currentAudioPage, setCurrentAudioPage] = useState(1);
    const ITEMS_PER_PAGE = 10;
    // grouped views (by filename) will be computed below
    async function fetchLists() {
        try {
            const v = await axios.get(`${API_BASE}/videos`);
            setVideos(v.data);
            setCurrentVideoPage(1);
            setActiveVideoTab(0);
        } catch (err) { console.error(err); }
        try {
            const a = await axios.get(`${API_BASE}/transcriptions`);
            setAudios(a.data);
            setCurrentAudioPage(1);
            setActiveAudioTab(0);
        } catch (err) { console.error(err); }
    }

    async function runImageSimilarity(db_id) {
        if (!db_id) return;
        try {
            const res = await axios.get(`${API_BASE}/search`, { params: { db_id, top_k: 5 } });
            setSimilarityModal({ open: true, type: 'image', query: db_id, matches: res.data.results || [] });
        } catch (err) {
            console.error('similarity error', err);
            setSimilarityModal({ open: true, type: 'image', query: db_id, matches: [] });
        }
    }

    async function runTextSimilarity(text) {
        try {
            const res = await axios.get(`${API_BASE}/search`, { params: { q: text, similar_text: true, top_k: 5 } });
            setSimilarityModal({ open: true, type: 'text', query: text, matches: res.data.results || [] });
        } catch (err) {
            console.error('similarity error', err);
            setSimilarityModal({ open: true, type: 'text', query: text, matches: [] });
        }
    }

    // Helper to group rows by filename into file-level entries
    function groupByFilename(rows) {
        // If the rows are already grouped by the backend (each item has a `rows` array), return as-is
        if (Array.isArray(rows) && rows.length > 0 && rows[0].rows) {
            return rows;
        }

        const map = {};
        for (const r of rows) {
            const name = r.filename || 'unknown';
            if (!map[name]) map[name] = { filename: name, rows: [] };
            map[name].rows.push(r);
        }
        return Object.values(map);
    }

    // Build a consolidated video object for a file group
    function buildVideoFromGroup(group) {
        // Return a consolidated object that keeps rows and keyframes from backend
        return {
            filename: group.filename,
            id: group.rows[0]?.id || null,
            rows: group.rows,
            keyframes: group.keyframes || []
        };
    }

    // Build a consolidated audio object for a file group
    function buildAudioFromGroup(group) {
        // Preserve per-row transcriptions instead of combining them
        return {
            filename: group.filename,
            id: group.rows[0]?.id || null,
            rows: group.rows
        };
    }

    async function handleSearch(query, page = 1) {
        if (!query.trim()) {
            setSearchResults([]);
            setSearchPagination(null);
            setHasSearched(false);
            setCurrentSearchPage(1);
            return;
        }

        setIsSearching(true);
        setHasSearched(true);
        try {
            const res = await axios.get(`${API_BASE}/search`, { params: { q: query, page: page, per_page: 10 } });
            setSearchResults(res.data.results);
            setSearchPagination(res.data.pagination);
            setCurrentSearchPage(page);
        } catch (err) {
            console.error('search error', err);
            setSearchResults([]);
            setSearchPagination(null);
        } finally {
            setIsSearching(false);
        }
    }

    // Calculate pagination for videos (grouped by filename)
    const videoGroups = groupByFilename(videos);
    const totalVideoPages = Math.ceil(videoGroups.length / ITEMS_PER_PAGE);
    const videoPaginationData = {
        total_count: videoGroups.length,
        total_pages: totalVideoPages,
        page: currentVideoPage,
        per_page: ITEMS_PER_PAGE
    };
    const videoStartIdx = (currentVideoPage - 1) * ITEMS_PER_PAGE;
    const videoEndIdx = videoStartIdx + ITEMS_PER_PAGE;
    const paginatedVideoGroups = videoGroups.slice(videoStartIdx, videoEndIdx);

    // Calculate pagination for audios (grouped by filename)
    const audioGroups = groupByFilename(audios);
    const totalAudioPages = Math.ceil(audioGroups.length / ITEMS_PER_PAGE);
    const audioPaginationData = {
        total_count: audioGroups.length,
        total_pages: totalAudioPages,
        page: currentAudioPage,
        per_page: ITEMS_PER_PAGE
    };
    const audioStartIdx = (currentAudioPage - 1) * ITEMS_PER_PAGE;
    const audioEndIdx = audioStartIdx + ITEMS_PER_PAGE;
    const paginatedAudioGroups = audioGroups.slice(audioStartIdx, audioEndIdx);

    // Separate search results into videos and audios
    const searchVideos = searchResults.filter(r => r.type === 'video');
    const searchAudios = searchResults.filter(r => r.type === 'audio');

    // Get current video for tab view
    const currentVideo = hasSearched ? null : (paginatedVideoGroups.length > 0 ? buildVideoFromGroup(paginatedVideoGroups[activeVideoTab] || paginatedVideoGroups[0]) : null);
    const currentAudio = hasSearched ? null : (paginatedAudioGroups.length > 0 ? buildAudioFromGroup(paginatedAudioGroups[activeAudioTab] || paginatedAudioGroups[0]) : null);

    // Ensure active tab indices stay in-bounds when pagination or data changes
    useEffect(() => {
        if (activeVideoTab >= paginatedVideoGroups.length) {
            setActiveVideoTab(Math.max(0, paginatedVideoGroups.length - 1));
        }
    }, [paginatedVideoGroups.length]);

    useEffect(() => {
        if (activeAudioTab >= paginatedAudioGroups.length) {
            setActiveAudioTab(Math.max(0, paginatedAudioGroups.length - 1));
        }
    }, [paginatedAudioGroups.length]);

    useEffect(() => { fetchLists(); }, []);

    return (
        <div className="app">
            {similarityModal.open && (
                <div className="similarity-modal" onClick={() => setSimilarityModal({ open: false, type: null, query: null, matches: [] })}>
                    <div className="similarity-modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <strong>Similar {similarityModal.type === 'image' ? 'Images' : 'Text'}</strong>
                            <button onClick={() => setSimilarityModal({ open: false, type: null, query: null, matches: [] })}>Close</button>
                        </div>
                        <div className="modal-body">
                            {similarityModal.matches.length === 0 ? (
                                <div>No similar items found</div>
                            ) : (
                                <div className="search-results">
                                    {similarityModal.matches.map((m, idx) => (
                                        <SearchSimilarityResult key={idx} result={m} onSimilarImage={runImageSimilarity} onSimilarText={runTextSimilarity} />
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
            <header className="app-header">
                <h1>HTX Multimedia Processor</h1>
                <p>Upload and process your media files</p>
            </header>

            <section className="process-section">
                <h2>Process Media Files</h2>
                <UploadForm onUploaded={fetchLists} />
            </section>

            <SearchSection
                isSearching={isSearching}
                onSearch={handleSearch} />

            <div className="content-container">
                <section className="media-section videos-section">
                    <h2>📹 Processed Videos</h2>

                    {hasSearched ? (
                        <div className="search-results">
                            {searchVideos.length === 0 ? (
                                <p className="empty-state">No videos found</p>
                            ) : (
                                searchVideos.map(v => <SearchVideoResult key={v.id} video={v} onSimilarImage={runImageSimilarity} />)
                            )}
                        </div>
                    ) : (
                        <>
                            {videos.length === 0 ? (
                                <p className="empty-state">No videos processed yet</p>
                            ) : (
                                <>
                                    {paginatedVideoGroups.length > 0 && (
                                        <>
                                            <div className="video-tabs">
                                                {paginatedVideoGroups.map((group, idx) => (
                                                    <button
                                                        key={group.filename}
                                                        className={`tab-button ${activeVideoTab === idx ? 'active' : ''}`}
                                                        onClick={() => { setActiveVideoTab(idx); setCurrentVideoPage(1); }}
                                                    >
                                                        {group.filename}
                                                    </button>
                                                ))}
                                            </div>
                                            <div className="tab-content">
                                                {currentVideo && <VideoCard key={currentVideo.filename || currentVideo.id} video={currentVideo} onSimilarImage={runImageSimilarity} />}
                                            </div>
                                        </>
                                    )}
                                    <PaginationControls
                                        pagination={videoPaginationData}
                                        currentPage={currentVideoPage}
                                        onPageChange={setCurrentVideoPage}
                                        isLoading={false} />
                                </>
                            )}
                        </>
                    )}
                </section>

                <section className="media-section audios-section">
                    <h2>🎵 Processed Audios</h2>
                    {hasSearched ? (
                        <div className="search-results">
                            {searchAudios.length === 0 ? (
                                <p className="empty-state">No audios found</p>
                            ) : (
                                searchAudios.map(a => <SearchAudioResult key={a.id} audio={a} onSimilarText={runTextSimilarity} />)
                            )}
                        </div>
                    ) : (
                        <>
                            {audios.length === 0 ? (
                                <p className="empty-state">No audios processed yet</p>
                            ) : (
                                <>
                                    {paginatedAudioGroups.length > 0 && (
                                        <>
                                            <div className="audio-tabs">
                                                {paginatedAudioGroups.map((group, idx) => (
                                                    <button
                                                        key={group.filename}
                                                        className={`tab-button ${activeAudioTab === idx ? 'active' : ''}`}
                                                        onClick={() => { setActiveAudioTab(idx); setCurrentAudioPage(1); }}
                                                    >
                                                        {group.filename}
                                                    </button>
                                                ))}
                                            </div>
                                            <div className="tab-content">
                                                {currentAudio && <AudioCard key={currentAudio.filename || currentAudio.id} audio={currentAudio} onSimilarText={runTextSimilarity} />}
                                            </div>
                                        </>
                                    )}
                                    <PaginationControls
                                        pagination={audioPaginationData}
                                        currentPage={currentAudioPage}
                                        onPageChange={setCurrentAudioPage}
                                        isLoading={false} />
                                </>
                            )}
                        </>
                    )}
                </section>
            </div>
        </div>
    );
}
