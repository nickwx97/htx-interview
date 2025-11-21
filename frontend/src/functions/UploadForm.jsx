import React, { useState } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE

export default function UploadForm({ onUploaded }) {
    const [files, setFiles] = useState(null)
    const [mediaType, setMediaType] = useState('video')
    const [uploading, setUploading] = useState(false)

    async function handleSubmit(e) {
        e.preventDefault()
        if (!files || files.length === 0) return
        setUploading(true)
        for (const f of files) {
            const form = new FormData()
            form.append('file', f)
            try {
                const url = `${API_BASE}/process/${mediaType}`
                const res = await axios.post(url, form, { headers: { 'Content-Type': 'multipart/form-data' } })
                console.log('uploaded', res.data)
            } catch (err) {
                console.error('upload error', err)
            }
        }
        setUploading(false)
        onUploaded()
    }

    return (
        <form onSubmit={handleSubmit} className="upload-form">
            <div>
                <label>Media Type: </label>
                <select value={mediaType} onChange={e => setMediaType(e.target.value)}>
                    <option value="video">Video</option>
                    <option value="audio">Audio</option>
                </select>
            </div>
            <div>
                <input type="file" multiple onChange={e => setFiles(e.target.files)} accept="video/*,audio/*" />
            </div>
            <div>
                <button type="submit" disabled={uploading}>{uploading ? 'Uploading...' : 'Upload'}</button>
            </div>
        </form>
    )
}