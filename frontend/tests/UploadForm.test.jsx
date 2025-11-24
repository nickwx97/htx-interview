// Tests for the upload form: upload flow and job polling
import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import UploadForm from '../src/functions/UploadForm.jsx'
import axios from 'axios'

vi.mock('axios')

describe('UploadForm', () => {
  beforeEach(() => {
    axios.post.mockReset()
    axios.get.mockReset()
  })

  it('uploads files and polls job status', async () => {
    const fakeJobs = { jobs: [{ job_id: 123, filename: 'test.mp4', status: 'queued' }] }
    axios.post.mockResolvedValueOnce({ data: fakeJobs })
    axios.get.mockResolvedValue({ data: { status: 'done', progress: 100 } })

    const onUploaded = vi.fn()
    render(<UploadForm onUploaded={onUploaded} />)

    const file = new File(['dummy content'], 'test.mp4', { type: 'video/mp4' })
    // find the file input element directly
    const fileInput = document.querySelector('input[type="file"]')
    expect(fileInput).toBeTruthy()

    fireEvent.change(fileInput, { target: { files: [file] } })

    const uploadBtn = screen.getByRole('button', { name: /upload/i })
    fireEvent.click(uploadBtn)

    await waitFor(() => expect(onUploaded).toHaveBeenCalled(), { timeout: 5000 })
  })
})
