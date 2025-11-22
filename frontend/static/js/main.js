// Text2SQL Frontend JavaScript

// DOM Elements
const questionInput = document.getElementById('question-input');
const dbSelect = document.getElementById('db-select');
const generateBtn = document.getElementById('generate-btn');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const sqlOutput = document.getElementById('sql-output');
const schemaOutput = document.getElementById('schema-output');
const examplesOutput = document.getElementById('examples-output');
const metaDb = document.getElementById('meta-db');
const metaExamples = document.getElementById('meta-examples');
const errorMessage = document.getElementById('error-message');
const copySqlBtn = document.getElementById('copy-sql-btn');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');

// Event Listeners
generateBtn.addEventListener('click', handleGenerate);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        handleGenerate();
    }
});
copySqlBtn.addEventListener('click', copyToClipboard);

// Main generation handler
async function handleGenerate() {
    const question = questionInput.value.trim();
    const dbId = dbSelect.value;

    if (!question) {
        showError('Please enter a question');
        return;
    }

    // Show loading state
    setLoading(true);
    hideError();
    hideResults();

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                db_id: dbId || null
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error: ${response.status}`);
        }

        displayResults(data);
    } catch (error) {
        showError(`Failed to generate SQL: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    // Generated SQL
    sqlOutput.textContent = data.generated_sql || 'No SQL generated';

    // Parse retrieved_context
    const context = data.retrieved_context || [];
    const schema = context.find(item => item.type === 'schema');
    const examples = context.filter(item => item.type === 'example');

    // Schema
    if (schema && schema.db_id) {
        schemaOutput.textContent = `Database: ${schema.db_id}\nFull schema retrieved from FAISS`;
    } else {
        schemaOutput.textContent = 'No schema retrieved';
    }

    // Examples
    if (examples.length > 0) {
        examplesOutput.innerHTML = examples.map(example => `
            <div class="example-item">
                <div class="example-question">Q: ${escapeHtml(example.question || 'N/A')}</div>
                <div class="example-sql">${escapeHtml(example.sql || 'N/A')}</div>
            </div>
        `).join('');
    } else {
        examplesOutput.innerHTML = '<p>No examples retrieved</p>';
    }

    // Metadata
    metaDb.textContent = data.db_id || 'Not specified';
    metaExamples.textContent = examples.length;

    // Show results
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format schema for display
function formatSchema(schema) {
    if (typeof schema === 'string') {
        return schema;
    }

    if (typeof schema === 'object') {
        return JSON.stringify(schema, null, 2);
    }

    return 'Schema format not recognized';
}

// Copy SQL to clipboard
function copyToClipboard() {
    const sql = sqlOutput.textContent;

    navigator.clipboard.writeText(sql).then(() => {
        const originalText = copySqlBtn.textContent;
        copySqlBtn.textContent = '✓ Copied!';
        copySqlBtn.style.background = '#28a745';
        copySqlBtn.style.color = 'white';

        setTimeout(() => {
            copySqlBtn.textContent = originalText;
            copySqlBtn.style.background = '';
            copySqlBtn.style.color = '';
        }, 2000);
    }).catch(err => {
        showError('Failed to copy to clipboard');
    });
}

// Show/hide functions
function setLoading(loading) {
    generateBtn.disabled = loading;
    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline';
    } else {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Check backend health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('System status:', data);

        if (data.backend === 'unreachable') {
            console.warn('Backend is not reachable. Some features may not work.');
        }
    } catch (error) {
        console.error('Failed to check system health:', error);
    }
});
