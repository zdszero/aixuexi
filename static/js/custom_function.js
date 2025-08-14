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
  if (localStorage.getItem('display-mode') === 'eyecare') {
    document.body.classList.remove('eyecare-mode')
  } else if (localStorage.getItem('display-mode') === 'dark') {
    document.body.classList.remove('dark-mode')
  } else {
    // do nothing, the default colorscheme is light
  }
}

function changePreBackground(color) {
  let pre_elements = document.querySelectorAll('pre')

  pre_elements.forEach((pre) => {
    pre.style.background = color
  })
}

function changeCodeBackground(color) {
  let code_elements = document.querySelectorAll('code')

  code_elements.forEach((code) => {
    code.style.background = color
  })
}

function changeIconColor(color) {
  let icons = document.getElementsByClassName('show-hide-icons')

  for (let i = 0; i < icons.length; i++) {
    icons[i].style.color = color
  }
}

function changeTableBackground(color1, color2, fontcolor) {
  table_elements = document.querySelectorAll('table')
  table_elements.forEach((table) => {
    table.style.setProperty('--bs-table-bg-type', color1)
    table.style.setProperty('--bs-table-striped-bg', color2)
    table.style.setProperty('--bs-table-color-state', fontcolor)
  })
}

// Utility function to parse RGB color string to array
function parseRgbColor(rgbString) {
  const result = rgbString.match(/\d+/g)
  return result ? [parseInt(result[0]), parseInt(result[1]), parseInt(result[2])] : [0, 0, 0]
}

function changeRects(strokeColor) {
  var rects = document.querySelectorAll('rect')
  rects.forEach((rect) => {
    // change stroke color
    var stroke = rect.getAttribute('stroke')
    if (stroke && stroke !== 'none') {
      rect.style.stroke = strokeColor;
    }
  })
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

// 加载明亮模式
function loadLightMode() {
  flip = shouldFlipRectFill('light')
  removeCurrentTheme()
  localStorage.setItem('display-mode', 'light')
  prism_css.href = "/css/prism.css"
  changePreBackground('#f8f8f8')
  changeIconColor('#333333')
  changeCodeBackground('#f8f9fa')
  changeTableBackground('white', '#f0f0f0', 'black')
  changeRects('black')
  if (flip) {
    flipColors()
  }
}

// 加载夜间模式
function loadDarkMode() {
  flip = shouldFlipRectFill('dark')
  removeCurrentTheme()
  document.body.classList.add('dark-mode')
  localStorage.setItem('display-mode', 'dark')
  prism_css.href = "/css/prism-dark.css"
  changePreBackground('#2d2d2d')
  changeIconColor('white')
  changeCodeBackground('#1b1f22')
  changeTableBackground('#2e2e2e', '#3c3c3c', 'white')
  changeRects('white')
  if (flip) {
    flipColors()
  }
}

// 加载护眼模式
function loadEyecareMode() {
  flip = shouldFlipRectFill('eyecare')
  removeCurrentTheme()
  document.body.classList.add('eyecare-mode')
  localStorage.setItem('display-mode', 'eyecare')
  prism_css.href = "/css/prism-eyecare.css"
  changePreBackground('#fdf6e3')
  changeIconColor('#333333')
  changeCodeBackground('#f8f9fa')
  changeTableBackground('#fff9c4', '#fff3e0', 'black')
  changeRects('black', flip)
  if (flip) {
    flipColors()
  }
}

// 初始化时检查 sidebar, toolbar, mode
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

  if (localStorage.getItem('display-mode') === 'eyecare') {
    loadEyecareMode()
  } else if (localStorage.getItem('display-mode') === 'dark') {
    loadDarkMode()
  } else {
    // do nothing, the default colorscheme is light
    // changePreBackground('#f8f8f8')
    // changeIconColor('#333333')
  }
})()

document.addEventListener('DOMContentLoaded', () => {
  localStorage.setItem('current-href', window.location.href)
  // 为所有的元素添加事件
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
