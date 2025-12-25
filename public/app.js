(() => {
  const API_BASE = '/api'
  
  // Theme
  const themeToggle = document.getElementById('themeToggle')
  const savedTheme = localStorage.getItem('theme') || 'light'
  
  function setTheme(theme){
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }

  setTheme(savedTheme)
  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme')
    setTheme(current === 'dark' ? 'light' : 'dark')
  })

  // Elements
  const packetsEl = document.getElementById('packets')
  const attacksEl = document.getElementById('attacks')
  const accuracyEl = document.getElementById('accuracy')
  const throughputEl = document.getElementById('throughput')
  const recentBody = document.getElementById('recentBody')
  const statusEl = document.getElementById('status')
  const startBtn = document.getElementById('startBtn')
  const stopBtn = document.getElementById('stopBtn')
  const injectBtn = document.getElementById('injectBtn')
  // New: UI for sending a test packet and showing result
  const resultBox = document.createElement('div')
  resultBox.id = 'resultBox'
  resultBox.style.margin = '16px 0 0 0'
  resultBox.style.fontWeight = 'bold'
  resultBox.style.color = '#2563eb'
  document.querySelector('.control-panel').appendChild(resultBox)

  let running = false
  let chartData = {
    allow: new Array(60).fill(0),
    block: new Array(60).fill(0),
    throttle: new Array(60).fill(0)
  }

  function addRecent(time, src, event, score){
    const tr = document.createElement('tr')
    tr.innerHTML = `<td>${time}</td><td>${src}</td><td>${event}</td><td>${score}</td>`
    recentBody.prepend(tr)
    while(recentBody.children.length > 8) recentBody.removeChild(recentBody.lastChild)
  }

  // API
  async function fetchStats(){
    try{
      const res = await fetch(`${API_BASE}/stats`)
      const data = await res.json()
      packetsEl.textContent = data.total_packets?.toLocaleString() || '--'
      attacksEl.textContent = data.detected_attacks || '--'
      accuracyEl.textContent = `${data.accuracy || '--'}%`
      throughputEl.textContent = `${data.throughput || '--'} pkts/s`
      // Update chartData if traffic_last_minute is present
      if(data.traffic_last_minute && typeof data.traffic_last_minute === 'object') {
        chartData = {
          allow: [...(data.traffic_last_minute.allow || new Array(60).fill(0))],
          block: [...(data.traffic_last_minute.block || new Array(60).fill(0))],
          throttle: [...(data.traffic_last_minute.throttle || new Array(60).fill(0))]
        }
        drawChart()
      }
    }catch(err){
      console.error('Stats error:', err)
    }
  }

  async function fetchLogs(){
    try{
      const res = await fetch(`${API_BASE}/logs?limit=8`)
      const logs = await res.json()
      recentBody.innerHTML = ''
      logs.forEach(log => {
        addRecent(log.timestamp || '--', log.source_ip || '--', log.attack_type || '--', log.confidence || '--')
      })
    }catch(err){
      console.error('Logs error:', err)
    }
  }

  const canvas = document.getElementById('trafficChart')
  const ctx = canvas.getContext('2d')

  function drawChart(){
    const w = canvas.clientWidth
    const h = canvas.height
    canvas.width = w
    ctx.clearRect(0, 0, w, h)

    // Chart colors for different actions
    const colors = {
      allow: 'rgba(16, 185, 129, 0.8)',   // Green
      block: 'rgba(239, 68, 68, 0.8)',    // Red
      throttle: 'rgba(245, 158, 11, 0.8)' // Yellow/Orange
    }

    const barWidth = w / 60 * 0.8  // 80% of available width per bar
    const barSpacing = w / 60 * 0.2  // 20% spacing
    const maxValue = Math.max(
      Math.max(...chartData.allow),
      Math.max(...chartData.block),
      Math.max(...chartData.throttle)
    ) || 1

    // Draw stacked bars for each second in chronological order
    // Newest data (index 59) on the left, oldest data (index 0) on the right
    for (let i = 0; i < 60; i++) {
      // Reverse the index: i=0 maps to dataIndex=59, i=59 maps to dataIndex=0
      const dataIndex = 59 - i
      const x = i * (w / 60) + barSpacing / 2
      const allowCount = chartData.allow[dataIndex] || 0
      const blockCount = chartData.block[dataIndex] || 0
      const throttleCount = chartData.throttle[dataIndex] || 0
      const totalCount = allowCount + blockCount + throttleCount

      if (totalCount === 0) continue

      // Calculate heights
      const allowHeight = (allowCount / maxValue) * (h - 20)
      const blockHeight = (blockCount / maxValue) * (h - 20)
      const throttleHeight = (throttleCount / maxValue) * (h - 20)

      // Draw stacked bars from bottom to top
      let currentY = h - 10

      // Draw allow (bottom)
      if (allowCount > 0) {
        ctx.fillStyle = colors.allow
        ctx.fillRect(x, currentY - allowHeight, barWidth, allowHeight)
        currentY -= allowHeight
      }

      // Draw block (middle)
      if (blockCount > 0) {
        ctx.fillStyle = colors.block
        ctx.fillRect(x, currentY - blockHeight, barWidth, blockHeight)
        currentY -= blockHeight
      }

      // Draw throttle (top)
      if (throttleCount > 0) {
        ctx.fillStyle = colors.throttle
        ctx.fillRect(x, currentY - throttleHeight, barWidth, throttleHeight)
      }
    }

    // Draw legend
    const legendY = 20
    const legendX = w - 150
    const legendItemWidth = 40

    ctx.fillStyle = colors.allow
    ctx.fillRect(legendX, legendY, 12, 12)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#000'
    ctx.font = '12px Segoe UI, sans-serif'
    ctx.fillText('Allow', legendX + 16, legendY + 10)

    // Block legend
    ctx.fillStyle = colors.block
    ctx.fillRect(legendX + legendItemWidth, legendY, 12, 12)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#000'
    ctx.fillText('Block', legendX + legendItemWidth + 16, legendY + 10)

    // Throttle legend
    ctx.fillStyle = colors.throttle
    ctx.fillRect(legendX + legendItemWidth * 2, legendY, 12, 12)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#000'
    ctx.fillText('Throttle', legendX + legendItemWidth * 2 + 16, legendY + 10)
  }

  // Controls
  startBtn.addEventListener('click', async () => {
    try{
      await fetch(`${API_BASE}/control/start`, { method: 'POST' })
      running = true
      statusEl.textContent = 'Running'
    }catch(err){
      console.error('Start error:', err)
    }
  })

  stopBtn.addEventListener('click', async () => {
    try{
      await fetch(`${API_BASE}/control/stop`, { method: 'POST' })
      running = false
      statusEl.textContent = 'Stopped'
    }catch(err){
      console.error('Stop error:', err)
    }
  })

  injectBtn.addEventListener('click', async () => {
    // Generate random features for demo (should match server expected length)
    const features = Array.from({length: 78}, () => Math.random())
    const source_ip = `10.0.0.${Math.floor(Math.random()*254+1)}`
    try{
      const res = await fetch(`${API_BASE}/packet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, source_ip })
      })
      if(res.ok){
        const data = await res.json()
        resultBox.textContent = `Detection: ${data.action} (Confidence: ${(data.confidence*100).toFixed(1)}%)`
        fetchLogs()
        await fetchStats()
        drawChart()
      }else{
        resultBox.textContent = 'Detection failed.'
      }
    }catch(err){
      resultBox.textContent = 'Error sending packet.'
      console.error('Inject error:', err)
    }
  })

  // Init
  drawChart()
  fetchStats()
  fetchLogs()

  // Polling
  setInterval(fetchStats, 5000)
  setInterval(fetchLogs, 10000)

  // Add test button for debugging
  const testBtn = document.createElement('button')
  testBtn.textContent = 'Test Chart'
  testBtn.style.position = 'fixed'
  testBtn.style.bottom = '10px'
  testBtn.style.right = '10px'
  testBtn.style.zIndex = '1000'
  testBtn.onclick = () => {
    // Fill with test data
    chartData = {
      allow: Array.from({length: 60}, () => Math.floor(Math.random() * 10)),
      block: Array.from({length: 60}, () => Math.floor(Math.random() * 5)),
      throttle: Array.from({length: 60}, () => Math.floor(Math.random() * 3))
    }
    console.log('Test data:', chartData)
    drawChart()
  }
  document.body.appendChild(testBtn)
})()
