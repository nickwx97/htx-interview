import React, { useState } from 'react'

// Simple pagination controls used across lists
export default function PaginationControls({ pagination, currentPage, onPageChange, isLoading }) {
    if (!pagination) return null

    return (
        <div className="pagination-controls">

            {pagination.total_pages > 1 && (
                <>
                    <button
                        onClick={() => onPageChange(currentPage - 1)}
                        disabled={currentPage === 1 || isLoading}
                        className="pagination-btn"
                    >
                        ← Previous
                    </button>

                    <div className="pagination-numbers">
                        {Array.from({ length: pagination.total_pages }, (_, i) => i + 1).map(pageNum => (
                            <button
                                key={pageNum}
                                onClick={() => onPageChange(pageNum)}
                                className={`page-number ${currentPage === pageNum ? 'active' : ''}`}
                                disabled={isLoading}
                            >
                                {pageNum}
                            </button>
                        ))}
                    </div>

                    <button
                        onClick={() => onPageChange(currentPage + 1)}
                        disabled={currentPage === pagination.total_pages || isLoading}
                        className="pagination-btn"
                    >
                        Next →
                    </button>
                </>
            )}
        </div>
    )
}