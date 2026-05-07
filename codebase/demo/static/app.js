document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const timeline = document.getElementById('timeline-container');
    const sourcesContainer = document.getElementById('sources-container');
    const knowledgeTitle = document.getElementById('knowledge-title');
    const knowledgeCount = document.getElementById('knowledge-count');
    const toast = document.getElementById('toast');

    let isWaitingForResponse = false;
    let activeCardIndex = -1;
    let messages = []; // store history in memory to handle citation clicks

    // --- UTILS --- //

    // Handle shift+enter for newlines
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        sendButton.disabled = input.value.trim() === '';
    });

    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), duration);
    }

    function scrollToBottom() {
        timeline.scrollTop = timeline.scrollHeight;
    }

    function parseInlineCitations(text) {
        // Replaces [1] with <span class="cite-num">1</span>
        return text.replace(/\[(\d+)\]/g, '<span class="cite-num" data-index="$1">$&</span>');
    }

    // --- RENDER TIMELINE CARDS --- //

    function renderUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg-card user';
        div.innerHTML = `
            <div class="icon">❓</div>
            <div class="content">${text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
        `;
        timeline.appendChild(div);
        scrollToBottom();
    }

    function renderLoadingSkeleton() {
        const div = document.createElement('div');
        div.className = 'msg-card loading';
        div.id = 'loading-card';
        div.innerHTML = `
            <div class="skeleton-bar"></div>
            <div class="skeleton-bar"></div>
            <div class="skeleton-bar"></div>
        `;
        timeline.appendChild(div);
        scrollToBottom();
    }

    function removeLoadingSkeleton() {
        const skeleton = document.getElementById('loading-card');
        if (skeleton) skeleton.remove();
    }

    function renderBotMessage(responseObj, index) {
        const div = document.createElement('div');
        
        let cardClass = 'msg-card bot';
        if (responseObj.action === 'low_confidence') cardClass += ' warning';
        if (responseObj.action === 'out_of_domain') cardClass += ' escalation';
        
        div.className = cardClass;
        div.dataset.index = index;

        let contentHtml = '';

        if (responseObj.action === 'out_of_domain') {
            contentHtml = `
                <div class="escalation-header">
                    <span>🛡 ESCALATED TO HUMAN AGENT</span>
                    <span class="ticket-id">${responseObj.ticket_id}</span>
                </div>
                <div class="content">
                    <p>This question falls outside my current knowledge domains. A support ticket has been created.</p>
                    <p style="margin-top: 12px;">Available knowledge domains:</p>
                    <ul>
                        ${responseObj.available_domains.map(d => `<li>${d.label} (${d.code.toUpperCase()})</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            if (responseObj.action === 'low_confidence') {
                contentHtml += `
                    <div class="warning-banner">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                           <path d="M12 2L1 21h22L12 2zm1 16h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                        </svg>
                        Limited confidence — verify with official sources
                    </div>
                `;
            }
            contentHtml += `<div class="content">${parseInlineCitations(responseObj.answer || "")}</div>`;
        }

        div.innerHTML = contentHtml;

        // Click to activate knowledge panel
        div.addEventListener('click', () => {
            activateBotCard(index);
        });

        timeline.appendChild(div);
        scrollToBottom();
        
        // Auto-activate this card to show sources immediately
        activateBotCard(index);
    }

    function renderErrorCard(message) {
        const div = document.createElement('div');
        div.className = 'msg-card error';
        div.innerHTML = `<div class="content">❌ ${message}</div>`;
        timeline.appendChild(div);
        scrollToBottom();
    }


    // --- RENDER KNOWLEDGE PANEL --- //

    function getRelevanceColor(score) {
        // Map typical cross-encoder scores (-10 to 10 approx) to colors
        if (score > 3) return 'var(--accent-emerald)';
        if (score > 0) return 'var(--accent-teal)';
        return 'var(--accent-amber)';
    }

    function getRelevancePercent(score) {
        // Very rough normalization for the UI bar (score normally ~ -5 to +10)
        let normalized = (score + 5) / 15 * 100;
        return Math.max(5, Math.min(100, normalized)) + '%';
    }

    function renderKnowledgePanel(responseObj) {
        sourcesContainer.innerHTML = ''; // clear

        if (!responseObj || responseObj.citations.length === 0) {
            knowledgeCount.textContent = '0 documents retrieved';
            sourcesContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📂</div>
                    <p>No source documents available for this response.</p>
                </div>
            `;
            return;
        }

        knowledgeCount.textContent = `${responseObj.citations.length} document${responseObj.citations.length > 1 ? 's' : ''} retrieved`;

        responseObj.citations.forEach((cite) => {
            const domainClass = cite.domain_code ? cite.domain_code.toLowerCase() : 'unknown';
            const div = document.createElement('div');
            div.className = 'source-card';
            
            div.innerHTML = `
                <div class="source-header">
                    <span class="source-index">[${cite.index}]</span>
                    <span class="source-title">📄 ${cite.title || 'Unknown Document'}</span>
                </div>
                <div class="source-meta">
                    <span class="domain-pill ${domainClass}">${cite.domain_code ? cite.domain_code.toUpperCase() : 'UNKNOWN'}</span>
                    <div class="relevance-container">
                        <span>Score: ${cite.score > 0 ? '+' : ''}${cite.score.toFixed(2)}</span>
                        <div class="relevance-bar-bg">
                            <div class="relevance-bar-fill" style="width: ${getRelevancePercent(cite.score)}; background-color: ${getRelevanceColor(cite.score)};"></div>
                        </div>
                    </div>
                </div>
                <div class="source-snippet">"${cite.snippet}"</div>
            `;
            sourcesContainer.appendChild(div);
        });
    }

    function activateBotCard(index) {
        // Remove active class from all
        document.querySelectorAll('.msg-card.bot').forEach(c => c.classList.remove('active'));
        
        // Add to selected
        const card = document.querySelector(`.msg-card.bot[data-index="${index}"]`);
        if (card) card.classList.add('active');

        // Update right panel
        const responseData = messages[index];
        if (responseData) {
            renderKnowledgePanel(responseData);
        }
    }


    // --- MAIN API CALL --- //

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query || isWaitingForResponse) return;

        // UI Prep
        isWaitingForResponse = true;
        input.value = '';
        input.style.height = 'auto';
        input.disabled = true;
        sendButton.disabled = true;

        renderUserMessage(query);
        renderLoadingSkeleton();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            removeLoadingSkeleton();

            if (!response.ok) {
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.error || 'Server error occurred');
            }

            const data = await response.json();
            
            // Store response data
            const index = messages.length;
            messages.push(data);
            
            renderBotMessage(data, index);

        } catch (error) {
            console.error('Error fetching chat API:', error);
            removeLoadingSkeleton();
            renderErrorCard(error.message || "Failed to connect to the server.");
            showToast("Server connection failed");
        } finally {
            isWaitingForResponse = false;
            input.disabled = false;
            input.focus();
        }
    });

    // --- QUICK QUESTIONS HANDLER --- //
    document.querySelectorAll('.quick-q').forEach(q => {
        q.addEventListener('click', () => {
            input.value = q.textContent;
            input.focus();
            // Automatically adjust the height of the textarea
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
            sendButton.disabled = input.value.trim() === '';
        });
    });
});
