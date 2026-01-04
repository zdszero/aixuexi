// Floating Action Button Tool Set JavaScript
(function() {
  'use strict';

  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', initFAB);

  function initFAB() {
    const fabContainer = document.getElementById('fab-container');
    const fabMain = document.getElementById('fab-main');
    const fabOptions = document.querySelectorAll('.fab-option');
    const tooltip = document.getElementById('fab-tooltip');
    
    if (!fabContainer || !fabMain) return;

    let isDragging = false;
    let currentX;
    let currentY;
    let initialX;
    let initialY;
    let xOffset = 0;
    let yOffset = 0;
    let hasMoved = false;
    let dragDistance = 0;
    const dragThreshold = 10; // pixels

    // Initialize position from localStorage if available
    const savedPosition = localStorage.getItem('fabPosition');
    if (savedPosition) {
      const pos = JSON.parse(savedPosition);
      fabContainer.style.right = 'auto';
      fabContainer.style.bottom = 'auto';
      fabContainer.style.left = pos.x + 'px';
      fabContainer.style.top = pos.y + 'px';
      xOffset = pos.x;
      yOffset = pos.y;
    }

    // Apply saved navbar pin state on page load
    applySavedNavbarState();

    // Toggle FAB menu with improved click detection
    fabMain.addEventListener('click', function(e) {
      // Only toggle if drag distance is minimal
      if (dragDistance < dragThreshold) {
        e.stopPropagation();
        toggleFAB();
      }
      dragDistance = 0;
      hasMoved = false;
    });

    // Close FAB when clicking outside
    document.addEventListener('click', function(e) {
      if (!fabContainer.contains(e.target) && fabContainer.classList.contains('active')) {
        fabContainer.classList.remove('active');
        removeBackdrop();
      }
    });

    // Prevent context menu on long press for drag
    fabMain.addEventListener('contextmenu', (e) => {
      if (isDragging) e.preventDefault();
    });

    // Drag functionality with improved detection
    fabMain.addEventListener('mousedown', dragStart);
    fabMain.addEventListener('touchstart', dragStart, { passive: false });
    document.addEventListener('mousemove', drag);
    document.addEventListener('touchmove', drag, { passive: false });
    document.addEventListener('mouseup', dragEnd);
    document.addEventListener('touchend', dragEnd);

    function dragStart(e) {
      dragDistance = 0;
      
      if (e.type === 'touchstart') {
        initialX = e.touches[0].clientX - xOffset;
        initialY = e.touches[0].clientY - yOffset;
      } else {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
      }

      if (e.target === fabMain || fabMain.contains(e.target)) {
        // Don't immediately set isDragging, wait for movement
        fabContainer.classList.add('dragging');
      }
    }

    function drag(e) {
      if (!fabContainer.classList.contains('dragging')) return;

      if (e.type === 'touchmove') {
        currentX = e.touches[0].clientX - initialX;
        currentY = e.touches[0].clientY - initialY;
      } else {
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
      }

      dragDistance = Math.sqrt(
        Math.pow(currentX - xOffset, 2) + Math.pow(currentY - yOffset, 2)
      );

      // Only start dragging if movement exceeds threshold
      if (dragDistance > dragThreshold) {
        e.preventDefault();
        isDragging = true;
        hasMoved = true;

        xOffset = currentX;
        yOffset = currentY;

        // Ensure FAB stays within viewport
        const rect = fabContainer.getBoundingClientRect();
        const maxX = window.innerWidth - rect.width;
        const maxY = window.innerHeight - rect.height;
        
        xOffset = Math.max(0, Math.min(xOffset, maxX));
        yOffset = Math.max(0, Math.min(yOffset, maxY));

        setTranslate(xOffset, yOffset, fabContainer);
      }
    }

    function setTranslate(xPos, yPos, el) {
      el.style.right = 'auto';
      el.style.bottom = 'auto';
      el.style.left = xPos + 'px';
      el.style.top = yPos + 'px';
    }

    function dragEnd(e) {
      if (isDragging) {
        initialX = currentX;
        initialY = currentY;
        isDragging = false;
        fabContainer.classList.remove('dragging');
        
        // Save position to localStorage
        localStorage.setItem('fabPosition', JSON.stringify({ x: xOffset, y: yOffset }));
        
        // Snap to edge if close
        snapToEdge();
      } else {
        // Reset dragging state even if minimal movement
        fabContainer.classList.remove('dragging');
      }
      dragDistance = 0;
    }

    function snapToEdge() {
      const rect = fabContainer.getBoundingClientRect();
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const threshold = 50;
      
      let newX = xOffset;
      let newY = yOffset;
      
      // Snap to left or right
      if (rect.left < threshold) {
        newX = 10;
      } else if (rect.right > windowWidth - threshold) {
        newX = windowWidth - rect.width - 10;
      }
      
      // Snap to top or bottom
      if (rect.top < threshold) {
        newY = 10;
      } else if (rect.bottom > windowHeight - threshold) {
        newY = windowHeight - rect.height - 10;
      }
      
      if (newX !== xOffset || newY !== yOffset) {
        xOffset = newX;
        yOffset = newY;
        fabContainer.style.transition = 'all 0.3s ease';
        setTranslate(xOffset, yOffset, fabContainer);
        localStorage.setItem('fabPosition', JSON.stringify({ x: xOffset, y: yOffset }));
        
        setTimeout(() => {
          fabContainer.style.transition = '';
        }, 300);
      }
    }

    function toggleFAB() {
      fabContainer.classList.toggle('active');
      if (fabContainer.classList.contains('active')) {
        createBackdrop();
        adjustLayoutDirection();
      } else {
        removeBackdrop();
      }
    }

    function adjustLayoutDirection() {
      const rect = fabContainer.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const windowWidth = window.innerWidth;
      
      // Circular layout works well everywhere, no special adjustments needed
      // Just ensure options are visible on screen
      ensureOptionsVisible();
    }
    
    function ensureOptionsVisible() {
      const options = fabContainer.querySelector('.fab-options');
      if (!options) return;
      
      const rect = options.getBoundingClientRect();
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const padding = 20;
      
      // For circular layout, options are positioned absolutely within FAB container
      // Check if the circle extends beyond viewport and adjust FAB position if needed
      
      if (rect.right > windowWidth - padding) {
        // FAB is too close to right edge, move it left
        const newX = Math.max(0, xOffset - (rect.right - (windowWidth - padding)));
        if (newX !== xOffset) {
          xOffset = newX;
          setTranslate(xOffset, yOffset, fabContainer);
        }
      }
      
      if (rect.left < padding) {
        // FAB is too close to left edge
        const newX = Math.min(window.innerWidth - fabContainer.getBoundingClientRect().width, xOffset + (padding - rect.left));
        if (newX !== xOffset) {
          xOffset = newX;
          setTranslate(xOffset, yOffset, fabContainer);
        }
      }
      
      if (rect.top < padding) {
        // FAB is too close to top
        const newY = Math.max(0, yOffset - (padding - rect.top));
        if (newY !== yOffset) {
          yOffset = newY;
          setTranslate(xOffset, yOffset, fabContainer);
        }
      }
      
      if (rect.bottom > windowHeight - padding) {
        // FAB is too close to bottom
        const newY = Math.min(window.innerHeight - fabContainer.getBoundingClientRect().height, yOffset + (rect.bottom - (windowHeight - padding)));
        if (newY !== yOffset) {
          yOffset = newY;
          setTranslate(xOffset, yOffset, fabContainer);
        }
      }
    }

    function createBackdrop() {
      if (!document.querySelector('.fab-backdrop')) {
        const backdrop = document.createElement('div');
        backdrop.className = 'fab-backdrop';
        document.body.appendChild(backdrop);
        setTimeout(() => backdrop.classList.add('active'), 10);
        
        backdrop.addEventListener('click', function(e) {
          e.stopPropagation();
          fabContainer.classList.remove('active');
          removeBackdrop();
        });
      }
    }

    function removeBackdrop() {
      const backdrop = document.querySelector('.fab-backdrop');
      if (backdrop) {
        backdrop.classList.remove('active');
        setTimeout(() => backdrop.remove(), 300);
      }
    }

    function showToast(message, type = 'info') {
      // Remove existing toast if any
      const existingToast = document.querySelector('.fab-toast');
      if (existingToast) {
        existingToast.remove();
      }

      // Create toast element
      const toast = document.createElement('div');
      toast.className = `fab-toast fab-toast-${type}`;
      toast.textContent = message;
      document.body.appendChild(toast);

      // Trigger animation
      setTimeout(() => toast.classList.add('show'), 10);

      // Remove after 2.5 seconds
      setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
      }, 2500);
    }

    function applySavedNavbarState() {
      const savedNavbarState = localStorage.getItem('navbarPinned');
      if (savedNavbarState === 'true') {
        // Apply pinned state but don't trigger the toggle animation
        pinNavbar(true);
      }
    }

    function togglePinNavbar() {
      const navbar = document.querySelector('.td-navbar');
      if (!navbar) return;

      const isCurrentlyFixed = getComputedStyle(navbar).position === 'fixed';
      
      if (isCurrentlyFixed) {
        // Unpin
        unpinNavbar();
        localStorage.setItem('navbarPinned', 'false');
        showToast('📌 导航栏已取消固定', 'success');
      } else {
        // Pin
        pinNavbar();
        localStorage.setItem('navbarPinned', 'true');
        showToast('📌 导航栏已固定', 'success');
      }
    }

    function pinNavbar(silent = false) {
      const navbar = document.querySelector('.td-navbar');
      const sidebarToc = document.querySelector('.td-sidebar-toc');
      const sidebarInner = document.querySelector('.td-sidebar__inner');
      const mainContent = document.querySelector('.td-main main, .td-404 main');
      
      if (!navbar) return;

      // Pin
      navbar.style.setProperty('position', 'fixed', 'important');
      navbar.style.top = '0';
      navbar.style.width = '100%';
      navbar.style.zIndex = '1020';

      const navbarHeight = navbar.offsetHeight;

      if (sidebarToc) {
        sidebarToc.style.setProperty('top', `${navbarHeight}px`, 'important');
        sidebarToc.style.setProperty('height', `calc(100vh - ${navbarHeight}px)`, 'important');
      }
      
      if (sidebarInner) {
        sidebarInner.style.setProperty('top', `${navbarHeight}px`, 'important')
        sidebarInner.style.height = `calc(100vh - ${navbarHeight + 16}px)`;
      }
      
      if (mainContent) {
        if (silent) {
          // For initial application, set immediately without transition
          mainContent.style.transition = 'none';
          mainContent.style.paddingTop = `${navbarHeight + 24}px`;
          // Force reflow
          mainContent.offsetHeight;
          mainContent.style.transition = '';
        } else {
          mainContent.style.paddingTop = `${navbarHeight + 24}px`;
        }
      }
    }

    function unpinNavbar() {
      const navbar = document.querySelector('.td-navbar');
      const sidebarToc = document.querySelector('.td-sidebar-toc');
      const sidebarInner = document.querySelector('.td-sidebar__inner');
      const mainContent = document.querySelector('.td-main main, .td-404 main');
      
      if (!navbar) return;

      // Unpin
      navbar.style.setProperty('position', 'relative', 'important');
      navbar.style.top = '0';

      if (sidebarToc) {
        sidebarToc.style.top = '0rem';
        sidebarToc.style.height = 'calc(100vh - 0rem)';
      }
      
      if (sidebarInner) {
        sidebarInner.style.top = '0rem';
        sidebarInner.style.height = 'calc(100vh - 1rem)';
      }
      
      if (mainContent) {
        mainContent.style.paddingTop = '1.5rem';
      }
    }

    function refreshChoices() {
      // First remove the stored choices
      try {
        const pageUrl = window.location.href;
        const storageKey = `quizChoices_${btoa(pageUrl)}`;
        localStorage.removeItem(storageKey);
      } catch (error) {
        console.error('Error clearing choice from localStorage:', error);
      }
      // Then refresh 
      location.reload();
    }

    function showAnswers() {
      const allChoices = document.querySelectorAll('.choice-container');
      allChoices.forEach(choice => {
        const toggleBtn = choice.querySelector('.toggle-btn');
        toggleBtn.click();
      });
      const allSolutions = document.querySelectorAll('.answer-container');
      allSolutions.forEach(choice => {
        const toggleBtn = choice.querySelector('.toggle-btn');
        toggleBtn.click();
      });
    }

    function openFeedback() {
        // Navigate to feedback page using assign method
        window.location.assign('/user_feedback/send_feedback');
    }

    function toggleValidateSelection() {
      let validateState;
      if (localStorage.getItem('validateSelection') === null) {
        validateState = false;
      } else {
        validateState = localStorage.getItem('validateSelection') === 'true';
      }
      
      const newState = !validateState;
      localStorage.setItem('validateSelection', newState.toString());
      
      if (newState) {
        showToast('✅ 选择验证已开启', 'success');
      } else {
        showToast('❌ 选择验证已关闭', 'info');
      }
      
      return newState;
    }

    function executeAction(action) {
      switch (action) {
        case 'toggleValidateSelection':
          toggleValidateSelection();
          break;

        case 'scrollTop':
          window.scrollTo({ top: 0, behavior: 'smooth' });
          break;

        case 'togglePinNavbar':
          togglePinNavbar();
          break;
        
        case 'refreshChoices':
          refreshChoices();
          break;
        
        case 'showAnswers':
          showAnswers();
          break;
        
        case 'openFeedback':
          openFeedback();
          break;
        
        default:
          console.log('Unknown action:', action);
      }
    }

    // Tool actions
    fabOptions.forEach(option => {
      option.addEventListener('click', function(e) {
        e.stopPropagation();
        const action = this.dataset.action;
        executeAction(action);
        
        // Close menu after action
        setTimeout(() => {
          fabContainer.classList.remove('active');
          removeBackdrop();
        }, 200);
      });

      // Tooltip on hover
      option.addEventListener('mouseenter', function(e) {
        const title = this.getAttribute('title');
        if (title && tooltip) {
          tooltip.textContent = title;
          tooltip.classList.add('show');
          const rect = this.getBoundingClientRect();
          tooltip.style.left = rect.left + rect.width / 2 + 'px';
          tooltip.style.top = rect.top - 35 + 'px';
        }
      });

      option.addEventListener('mouseleave', function() {
        if (tooltip) {
          tooltip.classList.remove('show');
        }
      });
    });

    // Initialize saved preferences
    if (localStorage.getItem('darkMode') === 'true') {
      document.body.classList.add('dark-mode');
      const icon = document.querySelector('[data-action="darkMode"] i');
      if (icon) icon.className = 'fas fa-sun';
    }

    const savedFontSize = localStorage.getItem('fontSize');
    if (savedFontSize) {
      document.documentElement.style.fontSize = {
        'small': '14px',
        'medium': '16px',
        'large': '18px',
        'extra-large': '20px'
      }[savedFontSize];
    }

    // Handle window resize
    window.addEventListener('resize', () => {
      const rect = fabContainer.getBoundingClientRect();
      const maxX = window.innerWidth - rect.width;
      const maxY = window.innerHeight - rect.height;
      
      if (xOffset > maxX) xOffset = maxX;
      if (yOffset > maxY) yOffset = maxY;
      
      setTranslate(xOffset, yOffset, fabContainer);
      localStorage.setItem('fabPosition', JSON.stringify({ x: xOffset, y: yOffset }));
      
      // Re-adjust layout on resize
      if (fabContainer.classList.contains('active')) {
        adjustLayoutDirection();
      }
    });
  }

  // Add necessary CSS animations
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideUp {
      from { transform: translateX(-50%) translateY(20px); opacity: 0; }
      to { transform: translateX(-50%) translateY(0); opacity: 1; }
    }
    @keyframes slideDown {
      from { transform: translateX(-50%) translateY(0); opacity: 1; }
      to { transform: translateX(-50%) translateY(20px); opacity: 0; }
    }
    
    /* Toast notification styles */
    .fab-toast {
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%) translateY(-100px);
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 14px 28px;
      border-radius: 50px;
      font-size: 15px;
      font-weight: 500;
      box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
      z-index: 10000;
      opacity: 0;
      transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
      pointer-events: none;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .fab-toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    
    .fab-toast-success {
      background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
      box-shadow: 0 8px 24px rgba(17, 153, 142, 0.4);
    }
    
    .fab-toast-info {
      background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
      box-shadow: 0 8px 24px rgba(52, 152, 219, 0.4);
    }
    
    .fab-toast-warning {
      background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
      box-shadow: 0 8px 24px rgba(243, 156, 18, 0.4);
    }
    
    .fab-toast-error {
      background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
      box-shadow: 0 8px 24px rgba(231, 76, 60, 0.4);
    }
    
    @keyframes toastSlide {
      0% {
        transform: translateX(-50%) translateY(-100px);
        opacity: 0;
      }
      100% {
        transform: translateX(-50%) translateY(0);
        opacity: 1;
      }
    }
  `;
  document.head.appendChild(style);
})();
