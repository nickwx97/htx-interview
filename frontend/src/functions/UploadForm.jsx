import React, { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE

export default function UploadForm({ onUploaded }) {
    const [filesList, setFilesList] = useState([]) // { file, progress, status, error }
    const [jobs, setJobs] = useState([]) // All jobs from backend
    const [page, setPage] = useState(1)
    const [total, setTotal] = useState(0)
    const [mediaType, setMediaType] = useState('video')
    const [uploading, setUploading] = useState(false)
    const perPage = 5

    function handleFilesChange(e) {
        const fs = Array.from(e.target.files || [])
        const list = fs.map(f => ({ file: f, progress: 0, status: 'pending', error: null }))
        setFilesList(list)
    }

    // Upload all files in a single multipart request and return server jobs
    async function uploadAllFiles() {
        const form = new FormData()
        filesList.forEach(f => form.append('files', f.file))
        const url = `${API_BASE}/process/${mediaType}`

        try {
            setFilesList(prev => prev.map(p => ({ ...p, status: 'uploading', progress: 0, error: null })))
            const res = await axios.post(url, form, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (ev) => {
                    const pct = ev.total ? Math.round((ev.loaded / ev.total) * 100) : 0
                    setFilesList(prev => prev.map(p => ({ ...p, progress: pct })))
                },
                timeout: 0,
            })
            return res.data
        } catch (err) {
            const msg = err?.response?.data?.detail || err.message || 'Upload failed'
            setFilesList(prev => prev.map(p => ({ ...p, status: 'error', error: String(msg) })))
            return null
        }
    }

    // Handle form submit: upload files then poll job statuses
    async function handleSubmit(e) {
        e.preventDefault()
        if (!filesList || filesList.length === 0) return
        setUploading(true)

        // Upload all files in one request and then poll jobs for processing status
        const resp = await uploadAllFiles()
        if (!resp || !resp.jobs) {
            setUploading(false)
            return
        }

        // Map returned jobs to files by filename and start polling
        const polls = resp.jobs.map(job => {
            const idx = filesList.findIndex(f => f.file.name === job.filename)
            if (idx === -1) return null
            // mark as queued
            setFilesList(prev => prev.map((p, i) => i === idx ? { ...p, status: 'queued', progress: 0 } : p))
            return pollJob(job.job_id, idx)
        }).filter(Boolean)

        // Wait for all polls to complete
        await Promise.all(polls)

        setUploading(false)
        onUploaded()
        setFilesList([])
        const fileInput = document.querySelector('.upload-form input[type="file"]')
        if (fileInput) fileInput.value = ''
    }


    // Poll a job until it's finished or errored; updates the file item in place
    async function pollJob(jobId, index) {
        const url = `${API_BASE}/process/status/${jobId}`
        try {
            while (true) {
                const r = await axios.get(url, { timeout: 0 })
                const data = r.data
                const status = data.status
                const progress = data.progress || 0
                setFilesList(prev => prev.map((p, i) => i === index ? { ...p, status, progress } : p))
                if (status === 'done' || status === 'error') {
                    return data
                }
                await new Promise(res => setTimeout(res, 1500))
            }
        } catch (err) {
            const msg = err?.response?.data?.detail || err.message || 'Polling failed'
            setFilesList(prev => prev.map((p, i) => i === index ? { ...p, status: 'error', error: String(msg) } : p))
            return null
        }
    }

    // Poll paginated jobs from backend every 5s (page-aware)
    useEffect(() => {
        let timer = null
        async function fetchJobs() {
            try {
                const res = await axios.get(`${API_BASE}/process/jobs?page=${page}&per_page=${perPage}`)
                if (res.data && res.data.jobs) {
                    setJobs(res.data.jobs)
                    setTotal(res.data.total || 0)
                }
            } catch (e) {}
        }
        fetchJobs()
        timer = setInterval(fetchJobs, 5000)
        return () => timer && clearInterval(timer)
    }, [page])

    return (
        <div>
            <form onSubmit={handleSubmit} className="upload-form">
                <div>
                    <label>Media Type: </label>
                    <select value={mediaType} onChange={e => setMediaType(e.target.value)}>
                        <option value="video">Video</option>
                        <option value="audio">Audio</option>
                    </select>
                </div>
                <div>
                    <input type="file" multiple onChange={handleFilesChange} accept="video/*,audio/*" />
                </div>
                <div className="upload-form-fullwidth">
                    {filesList.length > 0 && (
                        <div className="upload-list">
                            {filesList.map((it, idx) => (
                                <div key={idx} className="upload-item">
                                    <div className="upload-filename">{it.file.name}</div>
                                    <div className="upload-progress-row">
                                        <div className="progress-track">
                                            <div className={"progress-bar" + (it.status === 'error' ? ' error' : '')} style={{ width: `${it.progress}%` }} />
                                        </div>
                                        <div className="progress-meta">{it.progress}%</div>
                                    </div>
                                    <div className="upload-status">{it.status}{it.error ? ` — ${it.error}` : ''}</div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                <div>
                    <button type="submit" disabled={uploading || filesList.length === 0}>{uploading ? 'Uploading...' : 'Upload'}</button>
                </div>
            </form>

            {/* Persistent job list */}
            <div className="jobs-section">
                <h3>All Jobs (latest first)</h3>
                <div className="jobs-pagination-row">
                    <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1}>Prev</button>
                    <span className="page-info">Page {page} / {Math.max(1, Math.ceil((total || 0) / perPage))}</span>
                    <button onClick={() => setPage(p => p + 1)} disabled={page >= Math.ceil((total || 0) / perPage)}>Next</button>
                </div>
                <div className="upload-list">
                    {jobs.length === 0 && <div className="no-jobs-text">No jobs on this page.</div>}
                    {jobs.map(job => (
                        <div key={job.id} className="upload-item">
                            <div className="upload-filename">{job.filename}</div>
                            <div className="upload-progress-row">
                                <div className="progress-track">
                                    <div className={"progress-bar" + (job.status === 'error' ? ' error' : '')} style={{ width: `${job.progress}%` }} />
                                </div>
                                <div className="progress-meta">{job.progress}%</div>
                            </div>
                            <div className="upload-status">{job.status}{job.error ? ` — ${job.error}` : ''}</div>
                            <div className="upload-meta">{job.media_type} | {job.created_at && new Date(job.created_at).toLocaleString()}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}