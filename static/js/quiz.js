function getSubjectFromUrl(url) {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname.toLowerCase();
    
    // Define subject mapping rules based on URL patterns
    const subjectRules = [
      { pattern: /408quiz/i, subject: '408统考真题' },
      { pattern: /exercise/i, subject: '408练习题' },
      { pattern: /math_extra/i, subject: '数学模拟题' },
      { pattern: /math_old/i, subject: '早年数学真题' },
      { pattern: /math/i, subject: '数学真题' },
      { pattern: /english/i, subject: '英语真题' },
      { pattern: /politics/i, subject: '政治真题' },
    ];
    
    // Try to match against each rule
    for (const rule of subjectRules) {
      if (rule.pattern.test(pathname)) {
        return rule.subject;
      }
    }
    
    // Fallback: try to extract from path segments
    const segments = pathname.split('/').filter(s => s.length > 0);
    if (segments.length > 0) {
      const segment = segments[0];
      return segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, ' ');
    }
    
    return '未分类';
  } catch (error) {
    console.error('Error deducing subject from URL:', error);
    return '未分类';
  }
}

function showNotification(message) {
  const notification = document.createElement('div');
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #28a745;
    color: white;
    padding: 12px 24px;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
  `;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 2000);
}

