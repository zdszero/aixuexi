var show_sidebar_aside = document.getElementById('show-sidebar-col')
var sidebar = document.getElementsByClassName('td-sidebar')[0]
var show_toolbar_aside = document.getElementById('show-toolbar-col')
var toolbar = document.getElementsByClassName('td-sidebar-toc')[0]
var content_panel = document.getElementById('content-panel')
var cur_width = 2 / 3 * 100
var width_change = (1 / 6 - 0.1) * 100
var links = document.getElementsByTagName('link')
var prism_css

for (var i = 0; i < links.length; i++) {
  if (links[i].href.includes('prism.css')) {
    prism_css = links[i]
    break
  }
}

function hideSidebar() {
  sidebar.style.display = 'none'
  show_sidebar_aside.style.display = 'flex'
  cur_width += width_change
  content_panel.style.width = cur_width + '%'
}

function showSidebar() {
  sidebar.style.display = 'block'
  show_sidebar_aside.style.display = 'none'
  cur_width -= width_change
  content_panel.style.width = cur_width + '%'
  sidebar.style.visibility = 'visible'
}

function hideToolbar() {
  toolbar.setAttribute('style', 'display: none !important;')
  show_toolbar_aside.style.display = 'flex'
  cur_width += width_change
  content_panel.style.width = cur_width + '%'
}

function showToolbar() {
  toolbar.style.display = 'block'
  show_toolbar_aside.style.display = 'none'
  cur_width -= width_change
  content_panel.style.width = cur_width + '%'
  toolbar.style.visibility = 'visible'
}

function shouldFlipRectFill(nextMode) {
  prevHref = localStorage.getItem('current-href')
  displayMode = localStorage.getItem('display-mode')
  if (displayMode === null) {
    displayMode = "light"
  }
  if (prevHref !== window.location.href && displayMode == "dark") {
    return true
  }
  if (nextMode == "dark" && displayMode !== "dark") {
    return true
  } else if (nextMode !== "dark" && displayMode === "dark") {
    return true
  }
  return false
}

function removeCurrentTheme() {
  document.body.classList.remove('light-mode', 'eyecare-mode', 'dark-mode')
}

// Utility function to parse RGB color string to array
function parseRgbColor(rgbString) {
  const result = rgbString.match(/\d+/g)
  return result ? [parseInt(result[0]), parseInt(result[1]), parseInt(result[2])] : [0, 0, 0]
}

function flipColors() {
  function getInvertColor(color) {
      const rgbColor = color.startsWith('#') ? hexToRgb(color) : color
      const rgb = parseRgbColor(rgbColor)
      const invertedColor = `rgb(${255 - rgb[0]}, ${255 - rgb[1]}, ${255 - rgb[2]})`
      return invertedColor
  }
  const elements = document.querySelectorAll(
    '.svg-container rect, .svg-container path, .svg-container ellipse, .svg-container circle, .svg-container polygon, .svg-container polyline, .svg-container line'
  )
  curMode = localStorage.getItem('display-mode')
  elements.forEach((element) => {
    const currentColor = window.getComputedStyle(element).fill
    // invert color for background
    if (currentColor !== "none") {
      element.setAttribute('fill', getInvertColor(currentColor))
    }
    // stroke color as white or black
    var strokeColor = element.style.stroke || window.getComputedStyle(element).stroke;
    if (strokeColor !== "none") {
      toColor = 'black'
      if (curMode === 'dark') {
        toColor = 'white'
      }
      element.style.stroke = toColor;
    }
  })
}

// Load light mode
function loadLightMode() {
  flip = shouldFlipRectFill('light')
  removeCurrentTheme()
  document.body.classList.add('light-mode')
  localStorage.setItem('display-mode', 'light')
  prism_css.href = "/css/prism.css"
  if (flip) {
    flipColors()
  }
}

// Load dark mode
function loadDarkMode() {
  flip = shouldFlipRectFill('dark')
  removeCurrentTheme()
  document.body.classList.add('dark-mode')
  localStorage.setItem('display-mode', 'dark')
  prism_css.href = "/css/prism-dark.css"
  if (flip) {
    flipColors()
  }
}

// Load eyecare mode
function loadEyecareMode() {
  flip = shouldFlipRectFill('eyecare')
  removeCurrentTheme()
  document.body.classList.add('eyecare-mode')
  localStorage.setItem('display-mode', 'eyecare')
  prism_css.href = "/css/prism-eyecare.css"
  if (flip) {
    flipColors()
  }
}

// Initialize sidebar, toolbar, mode
(function() {
  if (localStorage.getItem('sidebar_hidden') == 1) {
    hideSidebar()
  } else {
    sidebar.style.visibility = 'visible'
  }
  if (localStorage.getItem('toolbar_hidden') == 1) {
    hideToolbar()
  } else {
    toolbar.style.visibility = 'visible'
  }
  
  // Set eyecare as default for new users
  if (!localStorage.getItem('display-mode')) {
    localStorage.setItem('display-mode', 'eyecare')
  }
  
  const displayMode = localStorage.getItem('display-mode');
  if (displayMode === 'eyecare') {
    loadEyecareMode()
  } else if (displayMode === 'dark') {
    loadDarkMode()
  } else {
    loadLightMode()
  }
})()

document.addEventListener('DOMContentLoaded', () => {
  localStorage.setItem('current-href', window.location.href)
  
  document.getElementById('hide_sidebar').addEventListener('click', function() {
    hideSidebar()
    localStorage.setItem('sidebar_hidden', 1)
  })

  show_sidebar_aside.addEventListener('click', function() {
    showSidebar()
    localStorage.setItem('sidebar_hidden', 0)
  })

  document.getElementById('hide_toolbar').addEventListener('click', function() {
    hideToolbar()
    localStorage.setItem('toolbar_hidden', 1)
  })

  show_toolbar_aside.addEventListener('click', function() {
    showToolbar()
    localStorage.setItem('toolbar_hidden', 0)
  })

  document.getElementById('toggle_light_mode').addEventListener('click', function(event) {
    event.preventDefault()
    if (localStorage.getItem('display-mode') == 'light') {
      return
    }
    loadLightMode()
  })
  
  document.getElementById('toggle_eyecare_mode').addEventListener('click', function(event) {
    event.preventDefault()
    if (localStorage.getItem('display-mode') == 'eyecare') {
      return
    }
    loadEyecareMode()
  })
  
  document.getElementById('toggle_dark_mode').addEventListener('click', function(event) {
    event.preventDefault()
    if (localStorage.getItem('display-mode') == 'dark') {
      return
    }
    loadDarkMode()
  })
})
