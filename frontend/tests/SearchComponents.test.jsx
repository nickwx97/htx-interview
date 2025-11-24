// Tests for search-related UI components
import React from 'react'
import { render, screen } from '@testing-library/react'
import { SearchVideoResult, SearchAudioResult, SearchSimilarityResult } from '../src/functions/Search.jsx'

describe('Search components', () => {
  it('renders video search result with keyframes and labels', () => {
    const video = {
      filename: 'vid.mp4',
      score: 2,
      snippet: 'some context here',
      keyframes: [{ url: '/uploads/v/1.jpg', timestamp: 0.5 }],
      detected_objects: [{ label: 'person', confidence: 0.8 }]
    }
    render(<SearchVideoResult video={video} />)
    expect(screen.getByText(/📹/)).toBeTruthy()
    expect(screen.getByText(/person/i)).toBeTruthy()
    expect(screen.getByText(/Keyframes:/i)).toBeTruthy()
  })

  it('renders audio search result with transcription', () => {
    const audio = { filename: 'aud.mp3', score: 1, snippet: 'hello world', transcriptions: 'hello world' }
    render(<SearchAudioResult audio={audio} />)
    expect(screen.getByText(/🎵/)).toBeTruthy()
    expect(screen.getByText(/Transcription:/i)).toBeTruthy()
    // The component may render the snippet and the transcription, so allow multiple matches
    const matches = screen.getAllByText(/hello world/i)
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('renders similarity results for video and audio', () => {
    const vres = { type: 'video', filename: 'v.mp4', score: 0.9, metadata: { url: '/uploads/v/kf_1.jpg' } }
    const ares = { type: 'audio', filename: 'a.mp3', score: 0.75, text: 'similar text' }
    render(<div>
      <SearchSimilarityResult result={vres} />
      <SearchSimilarityResult result={ares} />
    </div>)
    // There may be multiple similarity results rendered; assert at least one exists
    const sims = screen.getAllByText(/Similarity/i)
    expect(sims.length).toBeGreaterThanOrEqual(1)
  })
})
