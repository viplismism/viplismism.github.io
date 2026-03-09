---
layout: page
title: Papers
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>

<div id="paper-shelf"></div>
<div id="paper-modal-overlay"></div>

<style>
#paper-shelf {
  box-sizing: border-box !important;
}

#paper-shelf * {
  box-sizing: border-box !important;
}

#paper-shelf {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 16px !important;
  padding: 8px 0 16px 0 !important;
}

.paper-card {
  position: relative !important;
  width: 180px !important;
  height: 240px !important;
  border: 1.5px solid #d4d4d4 !important;
  cursor: pointer !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  transition: all 300ms ease !important;
  overflow: hidden !important;
  flex-shrink: 0 !important;
  background-color: #fff !important;
}

.paper-card:hover {
  border-color: #999 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}

.paper-card .paper-thumb {
  width: 100% !important;
  height: 160px !important;
  overflow: hidden !important;
  display: flex !important;
  align-items: flex-start !important;
  justify-content: center !important;
  border-bottom: 1px solid #eee !important;
  background-color: #fafafa !important;
}

.paper-card .paper-thumb canvas {
  width: 100% !important;
  height: auto !important;
  display: block !important;
}

.paper-card .paper-card-info {
  padding: 10px 12px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  flex: 1 !important;
  background-color: #fff !important;
}

.paper-card .paper-title {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  line-height: 1.35 !important;
  color: #222 !important;
  margin: 0 0 6px 0 !important;
  display: -webkit-box !important;
  -webkit-line-clamp: 2 !important;
  -webkit-box-orient: vertical !important;
  overflow: hidden !important;
}

.paper-card .paper-authors {
  font-size: 0.62rem !important;
  color: #999 !important;
  margin: 0 !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

#paper-modal-overlay {
  display: none;
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  z-index: 9999 !important;
  background: rgba(0,0,0,0.6) !important;
  backdrop-filter: blur(4px) !important;
  -webkit-backdrop-filter: blur(4px) !important;
}

#paper-modal-overlay.open {
  display: block !important;
}

.paper-modal {
  position: fixed !important;
  top: 3vh !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: 92vw !important;
  max-width: 780px !important;
  height: 94vh !important;
  background: #fff !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
}

.paper-modal-header {
  padding: 20px 24px 16px 24px !important;
  border-bottom: 1px solid #eee !important;
  flex-shrink: 0 !important;
  display: flex !important;
  justify-content: space-between !important;
  align-items: flex-start !important;
  gap: 16px !important;
  background: #fff !important;
}

.paper-modal-header-info h2 {
  margin: 0 0 4px 0 !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  color: #111 !important;
  line-height: 1.3 !important;
}

.paper-modal-header-info .modal-authors {
  color: #888 !important;
  font-size: 0.78rem !important;
  margin: 0 0 8px 0 !important;
  line-height: 1.4 !important;
}

.paper-modal-header-info .modal-links {
  display: flex !important;
  gap: 14px !important;
  align-items: center !important;
}

.paper-modal-header-info .modal-links a {
  font-size: 0.78rem !important;
  color: #0000FF !important;
  text-decoration: none !important;
  background: transparent !important;
}

.paper-modal-header-info .modal-links a:hover {
  text-decoration: underline !important;
}

.paper-modal-close {
  width: 32px !important;
  height: 32px !important;
  border: 1px solid #ddd !important;
  border-radius: 6px !important;
  background: #fff !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 1.1rem !important;
  color: #666 !important;
  flex-shrink: 0 !important;
  transition: all 200ms ease !important;
  padding: 0 !important;
  line-height: 1 !important;
}

.paper-modal-close:hover {
  border-color: #999 !important;
  color: #333 !important;
  background: #f5f5f5 !important;
}

.paper-modal-body {
  flex: 1 !important;
  overflow-y: auto !important;
  padding: 24px !important;
  background: #f5f5f5 !important;
  -webkit-overflow-scrolling: touch !important;
}

.paper-modal-body::-webkit-scrollbar {
  width: 6px !important;
}

.paper-modal-body::-webkit-scrollbar-track {
  background: transparent !important;
}

.paper-modal-body::-webkit-scrollbar-thumb {
  background: #ccc !important;
  border-radius: 3px !important;
}

.paper-modal-pages {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 12px !important;
}

.paper-modal-pages canvas {
  width: 100% !important;
  max-width: 700px !important;
  height: auto !important;
  display: block !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
  background: #fff !important;
}

.paper-modal-pages .loading-pages {
  color: #999 !important;
  font-size: 0.9rem !important;
  padding: 40px 0 !important;
  background: transparent !important;
}

@media (max-width: 600px) {
  .paper-card {
    width: calc(50% - 8px) !important;
    height: 200px !important;
  }

  .paper-card .paper-thumb {
    height: 120px !important;
  }

  .paper-modal {
    width: 100vw !important;
    height: 100vh !important;
    top: 0 !important;
    border-radius: 0 !important;
  }

  .paper-modal-header {
    padding: 14px 16px 12px 16px !important;
  }

  .paper-modal-body {
    padding: 12px !important;
  }
}
</style>

<script src="/assets/js/papers.js"></script>
