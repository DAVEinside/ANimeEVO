/**
 * Common JavaScript functions for Anime Character Evolution System
 */

// Wait for document to be ready
$(document).ready(function() {
    // Initialize theme toggle
    initThemeToggle();
    
    // Initialize global tooltips
    initTooltips();
    
    // Check GPU status
    checkGpuStatus();
});

/**
 * Initialize theme toggle (light/dark mode)
 */
function initThemeToggle() {
    const themeToggle = $('#theme-toggle');
    
    // Check stored theme or system preference
    const storedTheme = localStorage.getItem('theme');
    const systemDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Set initial theme
    if (storedTheme === 'dark' || (!storedTheme && systemDarkMode)) {
        document.body.classList.add('dark-mode');
        themeToggle.html('<i class="fas fa-sun"></i>');
    } else {
        document.body.classList.remove('dark-mode');
        themeToggle.html('<i class="fas fa-moon"></i>');
    }
    
    // Theme toggle click handler
    themeToggle.click(function() {
        if (document.body.classList.contains('dark-mode')) {
            // Switch to light mode
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
            themeToggle.html('<i class="fas fa-moon"></i>');
        } else {
            // Switch to dark mode
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
            themeToggle.html('<i class="fas fa-sun"></i>');
        }
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Check GPU status
 */
function checkGpuStatus() {
    $.get('/api/system/status', function(data) {
        if (data.success) {
            if (data.gpu_available) {
                $('#gpu-status').html('GPU: Active <i class="fas fa-check-circle text-success"></i>');
            } else {
                $('#gpu-status').html('GPU: Not Available <i class="fas fa-exclamation-circle text-warning"></i>');
            }
        } else {
            $('#gpu-status').html('GPU: Unknown <i class="fas fa-question-circle text-warning"></i>');
        }
    }).fail(function() {
        $('#gpu-status').html('GPU: Unknown <i class="fas fa-question-circle text-warning"></i>');
    });
}

/**
 * Show loading overlay
 * @param {string} message - Loading message to display
 */
function showLoading(message = "Processing...") {
    $('#loading-message').text(message);
    $('#loading-overlay').removeClass('d-none');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    $('#loading-overlay').addClass('d-none');
}

/**
 * Show success toast notification
 * @param {string} title - Toast title
 * @param {string} message - Toast message
 */
function showSuccess(title, message) {
    const toast = $('#toast-notification');
    toast.removeClass('bg-danger').addClass('bg-success');
    $('#toast-title').text(title);
    $('#toast-body').text(message);
    
    const bootstrapToast = new bootstrap.Toast(toast);
    bootstrapToast.show();
}

/**
 * Show error toast notification
 * @param {string} title - Toast title
 * @param {string} message - Toast message
 */
function showError(title, message) {
    const toast = $('#toast-notification');
    toast.removeClass('bg-success').addClass('bg-danger');
    $('#toast-title').text(title);
    $('#toast-body').text(message);
    
    const bootstrapToast = new bootstrap.Toast(toast);
    bootstrapToast.show();
}

/**
 * Format a date string
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date
 */
function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} length - Maximum length
 * @returns {string} Truncated text
 */
function truncateText(text, length = 100) {
    if (!text) return '';
    if (text.length <= length) return text;
    
    return text.substring(0, length) + '...';
}