// AGI Trinity - Main JavaScript

// Navigation active state
document.addEventListener('DOMContentLoaded', () => {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav a').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage || (currentPage === '' && href === 'index.html')) {
            link.classList.add('active');
        }
    });

    // Tab functionality
    initTabs();

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// Tab system
function initTabs() {
    document.querySelectorAll('.tabs').forEach(tabContainer => {
        const tabs = tabContainer.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetId = tab.dataset.tab;
                const parent = tabContainer.parentElement;

                // Update tab active state
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update content
                parent.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                const targetContent = parent.querySelector(`#${targetId}`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    });
}

// Copy to clipboard
function copyCode(button) {
    const codeBlock = button.parentElement.querySelector('code');
    navigator.clipboard.writeText(codeBlock.textContent).then(() => {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    });
}

// Agent info
const AGENTS = {
    claude: {
        name: 'Claude',
        model: 'Opus 4.5',
        role: '기술 전문가',
        color: '#f97316',
        strengths: ['코드 분석', '디버깅', '시스템 설계', '보안', '200K 컨텍스트']
    },
    gemini: {
        name: 'Gemini',
        model: '3 Pro',
        role: '데이터 분석가',
        color: '#3b82f6',
        strengths: ['리서치', '팩트체킹', '멀티모달', 'Deep Think 추론']
    },
    gpt: {
        name: 'GPT',
        model: '5.1',
        role: '창의적 문제해결사',
        color: '#10b981',
        strengths: ['창의적 솔루션', '브레인스토밍', '전략 수립', '통합 추론']
    }
};

// Strategy info
const STRATEGIES = {
    vote: {
        name: 'Vote',
        description: '최고 점수 응답 선택',
        useCase: '정답이 명확한 질문',
        icon: '🗳️'
    },
    synthesis: {
        name: 'Synthesis',
        description: '응답 통합',
        useCase: '종합적 분석 필요',
        icon: '🔄'
    },
    debate: {
        name: 'Debate',
        description: 'AI간 토론 후 결론',
        useCase: '논쟁적 주제',
        icon: '💬'
    },
    specialist: {
        name: 'Specialist',
        description: '전문가 자동 선택',
        useCase: '특화된 질문',
        icon: '🎯'
    }
};

// Export for use in other scripts
window.AGI = {
    AGENTS,
    STRATEGIES
};
