// Tests for UI card components (VideoCard, AudioCard)
import React from 'react'
import { render, screen } from '@testing-library/react'
import VideoCard from '../src/functions/VideoCard.jsx'
import AudioCard from '../src/functions/AudioCard.jsx'

describe('Card components', () => {
  it('renders VideoCard with keyframes and detected objects', () => {
    const video = {
      filename: 'vid.mp4',
      keyframes: [{ url: '/uploads/v/kf_1.jpg', timestamp: 0.5 }],
      rows: [{ frame_idx: 0, detected_objects: JSON.stringify([{ label: 'car', confidence: 0.7 }]) }]
    }
    render(<VideoCard video={video} />)
    expect(screen.getByText(/vid.mp4/i)).toBeTruthy()
    expect(screen.getByText(/Detected Objects/i) || screen.getByText(/No objects detected/i)).toBeTruthy()
  })

  it('renders AudioCard with rows and confidence badge', () => {
    const audio = {
      filename: 'aud.mp3',
      rows: [{ id: 1, transcriptions: 'hello', timestamps: JSON.stringify({ start: 0, end: 1, confidence: 0.85 }) }]
    }
    render(<AudioCard audio={audio} />)
    expect(screen.getByText(/aud.mp3/i)).toBeTruthy()
    expect(screen.getByText(/hello/i)).toBeTruthy()
    expect(screen.getByText(/85%/i)).toBeTruthy()
  })
})
